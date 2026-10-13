function out = gen_trajectory(varargin)
%GEN_TRAJECTORY  LEO pass + scatterer geometry for the ISAC channel emulator.
%
%   Emits traj.csv (header t_s,R0_m,G0_db,R1_m,G1_db,R2_m,G2_db) and
%   traj_meta.json into OutDir. Handoff section 5.
%
%   Three paths:
%     0  direct satellite, R0 = slant range
%     1  static clutter, a fixed scatterer near the receiver
%     2  moving target, a drone crossing the receiver
%
%   For a scatterer the excess path relative to the direct ray is
%       excess(t) = |r_tgt - r_rx| - u_sat(t) . (r_tgt - r_rx)
%   and R_p = R0 + excess. The satellite is effectively at infinity, so the
%   projection form is exact enough.
%
%   G_p is a plain echo-to-direct power ratio in dB, not a radar cross section.
%
%   Excess above the cyclic prefix puts a tap outside the CP where the
%   grid-based receiver cannot see it. Paths 0 and 1 are asserted to stay
%   inside for the whole run. Path 2 is a crossing event: a drone at 300 m
%   altitude flying 15 m/s for 600 s covers 9 km, and excess ~ |d| when the
%   satellite is high, so it can only be in the CP for a window around the
%   crossing. Outside that window G2 is driven to GainFloorDb, which both
%   models reality and gives section 10 its target-absent epochs for free.

    p = inputParser;
    p.addParameter('OutDir', fullfile(fileparts(mfilename('fullpath')),'..','data','traj'));
    p.addParameter('TlePath', '');
    p.addParameter('Lat', 37.3861);            % Mountain View
    p.addParameter('Lon', -122.0839);
    p.addParameter('AltM', 30);
    p.addParameter('Duration', 600);
    p.addParameter('Rate', 100);               % Hz
    p.addParameter('SearchStart', datetime(2026,9,7,0,0,0,'TimeZone','UTC'));
    p.addParameter('SearchHours', 24);
    p.addParameter('CarrierHz', 2.0e9);
    p.addParameter('CpLimitM', 1400);
    p.addParameter('CpMargin', 0.90);          % usable fraction of the CP
    p.addParameter('OccupiedBwHz', 4.5e6);     % sets the delay resolution cell
    p.addParameter('MinSepCells', 3);          % clutter/tap-0 separation floor
    p.addParameter('ClutterRangeM', 250);
    p.addParameter('ClutterAzDeg', 'auto');  % 'auto' maximises the worst-case excess
    p.addParameter('ClutterElDeg', 5);
    p.addParameter('ClutterGainDb', -20);
    p.addParameter('DroneAltM', 300);
    p.addParameter('DroneLateralM', 200);
    p.addParameter('DroneSpeedMps', 15);
    p.addParameter('DroneHeadingDeg', 0);      % 0 = due north
    p.addParameter('DroneGainDb', -25);
    p.addParameter('GainFloorDb', -200);       % "path absent"
    p.parse(varargin{:});
    a = p.Results;

    c = physconst('LightSpeed');
    od = a.OutDir; if ~exist(od,'dir'), mkdir(od); end

    tlePath = a.TlePath;
    if isempty(tlePath), tlePath = fullfile(od,'leo550.tle'); end
    if ~exist(tlePath,'file'), make_tle(tlePath); end

    % ---- find the highest-elevation pass in the search window ---------------
    scS = satelliteScenario(a.SearchStart, a.SearchStart+hours(a.SearchHours), 10);
    satS = satellite(scS, tlePath, 'OrbitPropagator','sgp4');
    % states() in R2023b returns the whole scenario timeline; it will not take
    % a time vector. The scenario SampleTime is therefore the sampling control.
    [rS, ~, tS] = states(satS, 'CoordinateFrame','ecef');
    [rGs, E, N, U] = gsFrame(a.Lat, a.Lon, a.AltM);
    elS = elevationDeg(rS, rGs, U);
    [maxEl, iTca] = max(elS);
    tTca = tS(iTca);
    assert(maxEl > 30, 'best pass only reaches %.1f deg elevation', maxEl);

    % ---- sample the pass at Rate Hz, centred on closest approach -----------
    tStart = tTca - seconds(a.Duration/2);
    tStop  = tTca + seconds(a.Duration/2);
    dt = 1/a.Rate;
    ts = (0:dt:a.Duration)';                          % seconds since tStart
    tv = tStart + seconds(ts);
    sc  = satelliteScenario(tStart, tStop, dt);
    sat = satellite(sc, tlePath, 'OrbitPropagator','sgp4');
    [r, v, tOut] = states(sat, 'CoordinateFrame','ecef');   % [3 x N] m, m/s
    N_t = numel(ts);
    assert(size(r,2) == N_t, 'states returned %d columns for %d times', size(r,2), N_t);
    assert(abs(seconds(tOut(end)-tOut(1)) - a.Duration) < 1e-6, 'timeline mismatch');

    d0 = r - rGs;                                     % [3 x N]
    R0 = vecnorm(d0, 2, 1)';                          % [N x 1]
    u  = d0 ./ R0';                                   % unit vector rx -> sat
    el = elevationDeg(r, rGs, U);
    assert(min(el) > 0, 'pass drops below the horizon (min el %.2f deg)', min(el));

    % ---- scatterers, in the local ENU frame ---------------------------------
    enu = @(e,n,up) rGs + E*e + N*n + U*up;

    elc = deg2rad(a.ClutterElDeg);
    clutterAt = @(azd) enu(a.ClutterRangeM*cos(elc)*sind(azd), ...
                           a.ClutterRangeM*cos(elc)*cosd(azd), ...
                           a.ClutterRangeM*sin(elc));
    if ischar(a.ClutterAzDeg) || isstring(a.ClutterAzDeg)
        % Two competing requirements. excess = |d|(1 - cos theta) collapses to
        % zero whenever the satellite passes through the clutter azimuth, which
        % merges the tap into tap 0 and makes it unresolvable. But the residual
        % Doppler that section 5 wants to demonstrate is d(excess)/dt, which is
        % bounded by (fc/c)*|d|*dtheta/dt and is zero for most azimuths.
        % Resolvability wins: maximise residual Doppler subject to a separation
        % floor, and fall back to pure separation if nothing clears the floor.
        floorM = a.MinSepCells*c/a.OccupiedBwHz;
        cand = 0:5:355; worst = zeros(size(cand)); rdpk = zeros(size(cand));
        for i = 1:numel(cand)
            e = excessPath(clutterAt(cand(i)), rGs, u);
            worst(i) = min(e);
            rdpk(i)  = max(abs(-a.CarrierHz/c*gradient(e, dt)));
        end
        okAz = worst >= floorM;
        if any(okAz)
            score = rdpk; score(~okAz) = -inf;
            [~, iBest] = max(score);
        else
            [~, iBest] = max(worst);
            warning('gen_trajectory:clutterSep', ...
                'no clutter azimuth clears %.0f m separation; best is %.0f m', ...
                floorM, max(worst));
        end
        clutterAz = cand(iBest);
        fprintf(['  clutter azimuth auto -> %d deg ' ...
                 '(worst-case excess %.0f m, peak residual Doppler %.2f Hz; ' ...
                 '%d/%d azimuths clear the %.0f m floor)\n'], ...
                clutterAz, worst(iBest), rdpk(iBest), nnz(okAz), numel(cand), floorM);
    else
        clutterAz = a.ClutterAzDeg;
    end
    r1 = clutterAt(clutterAz);
    [ex1, rng1] = excessPath(r1, rGs, u);

    hd = deg2rad(a.DroneHeadingDeg);
    along = a.DroneSpeedMps*(ts - a.Duration/2);      % 0 at mid-pass
    lat2  = a.DroneLateralM;
    eD = along*sin(hd) + lat2*cos(hd);
    nD = along*cos(hd) - lat2*sin(hd);
    r2 = rGs + E*eD' + N*nD' + U*repmat(a.DroneAltM,1,N_t);
    [ex2, rng2] = excessPath(r2, rGs, u);

    % ---- cyclic prefix budget ------------------------------------------------
    lim = a.CpLimitM*a.CpMargin;
    assert(max(ex1) < lim, ...
        'clutter excess %.1f m exceeds the %.1f m CP budget', max(ex1), lim);
    valid2 = ex2 < lim;
    assert(any(valid2), 'drone never enters the CP budget (min excess %.1f m)', min(ex2));
    [w0, w1] = longestRun(valid2);
    assert(ts(w1)-ts(w0) > 30, 'drone visible for only %.1f s', ts(w1)-ts(w0));
    valid2(:) = false; valid2(w0:w1) = true;          % single contiguous event

    R1 = R0 + ex1;
    R2 = R0 + ex2;
    G0 = zeros(N_t,1);
    G1 = repmat(a.ClutterGainDb, N_t, 1);
    G2 = repmat(a.GainFloorDb,  N_t, 1);  G2(valid2) = a.DroneGainDb;

    % ---- diagnostics ---------------------------------------------------------
    % Range rate straight from the propagated ECEF velocity: the ground station
    % is fixed in ECEF, so the relative velocity is the satellite's.
    rr0  = sum(u .* v, 1)';                           % m/s
    fd0  = -a.CarrierHz/c*rr0;                        % Hz, Doppler
    fdd0 = gradient(fd0, dt);                         % Hz/s, Doppler rate
    fs   = 7.68e6;
    walk = abs(rr0)/c*fs;                             % delay walk, samples/s
    res1 = -a.CarrierHz/c*gradient(ex1, dt);          % residual Doppler vs path 0
    res2 = -a.CarrierHz/c*gradient(ex2, dt);

    % ---- write ----------------------------------------------------------------
    csvPath = fullfile(od,'traj.csv');
    fid = fopen(csvPath,'w');
    fprintf(fid, 't_s,R0_m,G0_db,R1_m,G1_db,R2_m,G2_db\n');
    fprintf(fid, '%.6f,%.6f,%.4f,%.6f,%.4f,%.6f,%.4f\n', ...
            [ts, R0, G0, R1, G1, R2, G2]');
    fclose(fid);

    % R_nominal: handoff section 5 says "R0 at t = 0", but this pass starts at
    % low elevation, so R0(0) is the MAXIMUM range and every tau would come out
    % negative - a non-causal delay line. Any constant nominal is physically
    % equivalent (it is a fixed phase rotation and a fixed range offset, and it
    % leaves inter-path delays and all Doppler untouched), so use the minimum
    % range over every present path instead. That is the choice that keeps
    % tau >= 0. r0_at_t0_m is still reported for reference.
    Rall = [R0; R1; R2(valid2)];
    % traj.csv stores ranges at %.6f, so r_nominal must be quantised the same
    % way or a consumer that takes the minimum over the CSV disagrees with one
    % that reads this field. The gap is sub-micrometre, but fc/c amplifies it
    % into ~1e-5 of carrier phase, which is enough to make the C++ and Python
    % channels visibly diverge. Floor rather than round, so tau stays >= 0.
    rNominal = floor(min(Rall)*1e6)/1e6;
    tauMax = (max(Rall) - rNominal)/c;

    m = struct();
    m.carrier_hz   = a.CarrierHz;
    m.r_nominal_m  = rNominal;
    m.r0_at_t0_m   = R0(1);
    m.tau_max_s    = tauMax;
    m.tau_max_samples = tauMax*fs;
    m.duration_s   = a.Duration;
    m.rate_hz      = a.Rate;
    m.n_samples    = N_t;
    m.epoch_utc    = char(tStart, 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z''');
    m.tca_utc      = char(tTca,   'yyyy-MM-dd''T''HH:mm:ss.SSS''Z''');
    m.tle          = tlePath;
    m.ground_station = struct('lat_deg',a.Lat,'lon_deg',a.Lon,'alt_m',a.AltM);
    m.cp_limit_m   = a.CpLimitM;
    m.cp_budget_m  = lim;
    m.paths = { ...
      struct('id',0,'name','direct', ...
             'description','Direct satellite path; R0 is the slant range.', ...
             'gain_db',0,'excess_m_max',0,'always_present',true), ...
      struct('id',1,'name','clutter', ...
             'description',sprintf('Static scatterer %.0f m from the receiver at %.0f deg azimuth, %.0f deg elevation.', a.ClutterRangeM, clutterAz, a.ClutterElDeg), ...
             'gain_db',a.ClutterGainDb,'excess_m_max',max(ex1),'always_present',true), ...
      struct('id',2,'name','drone', ...
             'description',sprintf('Drone at %.0f m altitude, %.0f m lateral offset, crossing at %.0f m/s.', a.DroneAltM, a.DroneLateralM, a.DroneSpeedMps), ...
             'gain_db',a.DroneGainDb,'excess_m_max',max(ex2(valid2)),'always_present',false)};
    % A window is "well separated" when every present tap is at least one delay
    % resolution cell away from tap 0 and from the other tap. Outside it the
    % demo is looking at merged peaks, which is not a receiver failure but does
    % make a poor slide.
    cell_m = c/a.OccupiedBwHz;                        % delay resolution in metres
    sep = (ex1 > cell_m) & valid2 & (ex2 > cell_m) & (abs(ex2-ex1) > cell_m);
    wins = allRuns(sep);
    wins = wins(ts(wins(:,2)) - ts(wins(:,1)) > 30, :);
    m.well_separated_windows_s = arrayfun(@(k) [ts(wins(k,1)) ts(wins(k,2))], ...
                                          1:size(wins,1), 'UniformOutput', false);
    if isempty(m.well_separated_windows_s)
        m.well_separated_window_s = [];
    else
        [~, kL] = max(ts(wins(:,2)) - ts(wins(:,1)));
        m.well_separated_window_s = [ts(wins(kL,1)) ts(wins(kL,2))];
    end
    m.delay_resolution_m = cell_m;
    % A static scatterer at range |d| cannot show more residual Doppler than
    % (fc/c)*|d|*dtheta/dt, with dtheta/dt the line-of-sight rotation rate.
    losRate = max(vecnorm(gradient(u, dt), 2, 1));     % rad/s
    m.los_rotation_deg_s = rad2deg(losRate);
    m.clutter_residual_doppler_bound_hz = a.CarrierHz/c*a.ClutterRangeM*losRate;
    m.clutter_azimuth_deg = clutterAz;
    m.target_window_s   = [ts(w0) ts(w1)];
    m.target_absent_s   = {[ts(1) ts(w0)], [ts(w1) ts(end)]};
    m.elevation_deg     = struct('min',min(el),'max',max(el));
    m.slant_range_m     = struct('min',min(R0),'max',max(R0),'at_t0',R0(1));
    m.range_rate_mps    = struct('min',min(rr0),'max',max(rr0),'abs_max',max(abs(rr0)));
    m.doppler_hz        = struct('min',min(fd0),'max',max(fd0),'abs_max',max(abs(fd0)));
    m.doppler_rate_hz_s = struct('abs_max',max(abs(fdd0)));
    m.delay_walk_samples_per_s = struct('abs_max',max(walk),'fs_hz',fs);
    m.residual_doppler_hz = struct( ...
        'clutter_abs_max', max(abs(res1)), ...
        'drone_min', min(res2(valid2)), 'drone_max', max(res2(valid2)));
    m.excess_m = struct('clutter_min',min(ex1),'clutter_max',max(ex1), ...
                        'drone_min',min(ex2(valid2)),'drone_max',max(ex2(valid2)));
    m.range_m  = struct('clutter',rng1(1),'drone_min',min(rng2(valid2)),'drone_max',max(rng2(valid2)));
    fid = fopen(fullfile(od,'traj_meta.json'),'w');
    fprintf(fid,'%s\n', jsonencode(m,'PrettyPrint',true)); fclose(fid);

    fprintf('gen_trajectory -> %s\n', od);
    fprintf('  pass: max el %.1f deg, TCA %s, %.0f s at %g Hz (%d rows)\n', ...
        maxEl, string(tTca), a.Duration, a.Rate, N_t);
    fprintf('  slant range %.1f .. %.1f km, R_nominal %.1f km (R0 at t=0 was %.1f km)\n', ...
        min(R0)/1e3, max(R0)/1e3, rNominal/1e3, R0(1)/1e3);
    fprintf('  tau in [0, %.3f ms] = [0, %.0f samples] at %.2f MS/s\n', ...
        tauMax*1e3, tauMax*fs, fs/1e6);
    fprintf('  range rate +-%.0f m/s -> Doppler +-%.1f kHz, rate up to %.0f Hz/s\n', ...
        max(abs(rr0)), max(abs(fd0))/1e3, max(abs(fdd0)));
    fprintf('  delay walk up to %.0f samples/s at %.2f MS/s  (%.1f samples per 20 ms CPI)\n', ...
        max(walk), fs/1e6, max(walk)*0.020);
    fprintf('  clutter excess %.1f .. %.1f m (az %d deg), residual Doppler up to %.2f Hz\n', ...
        min(ex1), max(ex1), clutterAz, max(abs(res1)));
    fprintf('  LOS rotates at up to %.2f deg/s -> clutter residual Doppler bound %.1f Hz\n', ...
        m.los_rotation_deg_s, m.clutter_residual_doppler_bound_hz);
    for k = 1:numel(m.well_separated_windows_s)
        wk = m.well_separated_windows_s{k};
        fprintf('  all taps resolvable (>%.0f m apart) over t = %.1f .. %.1f s\n', ...
            cell_m, wk(1), wk(2));
    end
    fprintf('  drone visible %.1f .. %.1f s, excess %.1f .. %.1f m, residual Doppler %.1f .. %.1f Hz\n', ...
        ts(w0), ts(w1), min(ex2(valid2)), max(ex2(valid2)), min(res2(valid2)), max(res2(valid2)));

    out = struct('t',ts,'R0',R0,'R1',R1,'R2',R2,'G0',G0,'G1',G1,'G2',G2, ...
                 'el',el,'meta',m,'csv',csvPath);
end

% -------------------------------------------------------------------------
function [rGs, E, N, U] = gsFrame(lat, lon, alt)
    rGs = lla2ecef([lat lon alt])';                   % [3 x 1] metres
    la = deg2rad(lat); lo = deg2rad(lon);
    U = [cos(la)*cos(lo); cos(la)*sin(lo); sin(la)];
    E = [-sin(lo); cos(lo); 0];
    N = cross(U, E);
end

function el = elevationDeg(r, rGs, U)
    d = r - rGs;
    el = 90 - rad2deg(acos((U'*d) ./ vecnorm(d,2,1)))';
end

function [ex, rng] = excessPath(rTgt, rGs, u)
%EXCESSPATH  excess = |d| - u.d, with d the receiver-to-scatterer vector.
    d   = rTgt - rGs;                                 % [3 x 1] or [3 x N]
    rng = vecnorm(d, 2, 1)';
    ex  = rng - sum(u .* d, 1)';
end

function runs = allRuns(mask)
    mask = mask(:)'; d = diff([0 mask 0]);
    runs = [find(d == 1)', (find(d == -1) - 1)'];
end

function [i0, i1] = longestRun(mask)
    mask = mask(:)'; d = diff([0 mask 0]);
    s = find(d == 1); e = find(d == -1) - 1;
    [~, k] = max(e - s); i0 = s(k); i1 = e(k);
end
