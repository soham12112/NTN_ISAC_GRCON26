function out = gen_ntn_channel(varargin)
%GEN_NTN_CHANNEL  Add 3GPP TR 38.811 NTN multipath to a LEO trajectory.
%
%   Reads a trajectory produced by gen_trajectory and writes a new one with
%   the clusters of a 3GPP NTN tapped-delay-line profile layered on top, so
%   the GNU Radio channel block plays a standards-defined NTN channel without
%   any change to the block, the file contract, or the receiver.
%
%   Name-value arguments:
%     InDir      trajectory to extend            (default ../data/traj_ota)
%     OutDir     where to write                  (default <InDir>_ntn)
%     Profile    NTN-TDL-A | -B | -C | -D | NTN-TDLA100 | NTN-TDLC5
%                                                (default 'NTN-TDL-C')
%     DelaySpread   seconds, scales the profile  (default 100e-9)
%     MobileSpeed   m/s, sets the Doppler spread (default 0.83, 3 km/h)
%     SubRays    sub-rays per cluster            (default 1)
%     CarrierHz  carrier, for Doppler            (default from traj_meta)
%     MaxExcessM cluster excess-path budget      (default 1260, 0.9*CP)
%
%   Why this needs no new channel code
%   ---------------------------------
%   The obvious way to import a stochastic channel is to give each tap a
%   complex gain that varies with time, which means a Doppler per tap. The
%   channel block deliberately cannot express that: handoff section 6 says
%   delay is the only thing injected and Doppler must fall out of the carrier
%   phase, because a delay and a Doppler specified independently disagree and
%   any receiver that jointly tracks them sees physically impossible data.
%
%   So each cluster is emitted as a path whose RANGE RAMPS, at exactly the rate
%   whose implied Doppler is the one wanted:
%
%       R_p(t) = R_0(t) + c*tau_p - f_d,p * (c/fc) * t
%
%   Delay and Doppler then agree by construction, the existing traj.csv
%   contract carries it unchanged, and the C++ block needs no modification.
%   That is also how linear_path in py/ntn_isac/trajectory.py already works.
%
%   Fading within a cluster comes from splitting it into SubRays sub-rays with
%   different Dopplers drawn from the Jakes spectrum, which is what a cluster
%   is anyway - their sum beats at the difference frequencies and reproduces
%   the cluster's fading. SubRays=1 gives a clean Doppler-shifted tap with no
%   fading, which is the right default here because every extra path costs
%   real-time margin in the channel block. Measured throughput is about
%   20 MS/s at 3 paths and 8 interpolator taps against a 7.68 MS/s
%   requirement, and the cost is roughly linear in the path count, so the
%   budget is about 8 paths. The function prints its estimate.
%
%   On not double-counting the line of sight
%   ----------------------------------------
%   Tap 0 of an NTN-TDL profile is Rician with K = 13.3 dB, so it is the line
%   of sight plus a diffuse part. The trajectory's path 0 already IS the line
%   of sight. Adding tap 0 whole would count it twice, so only its diffuse
%   part is emitted, at -K dB relative to path 0, and every other cluster is
%   referenced to the same line-of-sight power.
%
%   Note what this model is and is not for. It adds realistic NTN multipath and
%   fading, which is what makes a communication measurement credible. It does
%   not add a target: its clusters are statistical, they have no ground truth,
%   and the sensing checkpoints must keep using the commanded scatterers of
%   gen_trajectory. Run with the clusters on to show the link survives a
%   standards-defined channel; score sensing against the deterministic paths.

    p = inputParser;
    root = fullfile(fileparts(mfilename('fullpath')), '..');
    p.addParameter('InDir', fullfile(root,'data','traj_ota'));
    p.addParameter('OutDir', '');
    p.addParameter('Profile', 'NTN-TDL-C');
    p.addParameter('DelaySpread', 100e-9);
    p.addParameter('MobileSpeed', 0.83);
    p.addParameter('SubRays', 1);
    p.addParameter('CarrierHz', 0);
    p.addParameter('MaxExcessM', 1260);
    p.addParameter('RefTime', 300);
    p.addParameter('Seed', 2026);
    p.parse(varargin{:});
    a = p.Results;

    c = physconst('LightSpeed');
    inDir = a.InDir;
    outDir = a.OutDir;
    if isempty(outDir), outDir = [inDir '_ntn']; end
    if ~exist(outDir,'dir'), mkdir(outDir); end

    meta = jsondecode(fileread(fullfile(inDir,'traj_meta.json')));
    fc = a.CarrierHz; if fc <= 0, fc = meta.carrier_hz; end

    % ---- read the base trajectory -------------------------------------------
    fid = fopen(fullfile(inDir,'traj.csv'));
    hdr = strsplit(strtrim(fgetl(fid)), ',');
    fclose(fid);
    M = readmatrix(fullfile(inDir,'traj.csv'), 'NumHeaderLines', 1);
    ts = M(:,1);
    nBase = (numel(hdr)-1)/2;
    R = M(:, 2:2:end);          % [N x nBase]
    G = M(:, 3:2:end);
    N = numel(ts);

    % ---- the 3GPP profile ---------------------------------------------------
    ch = nrTDLChannel;
    ch.DelayProfile   = a.Profile;
    ch.DelaySpread    = a.DelaySpread;
    % nrTDLChannel has no CarrierFrequency; the carrier enters only through
    % MaximumDopplerShift, which is computed here instead.
    ch.MaximumDopplerShift = a.MobileSpeed*fc/c;
    ch.SampleRate     = 7.68e6;
    ci = info(ch);
    tau = ci.PathDelays(:);              % s
    gdb = ci.AveragePathGains(:);        % dB, relative to the profile total
    Kdb = ci.KFactorFirstTap;            % dB, Rician K of tap 0
    fD  = ch.MaximumDopplerShift;

    % Split tap 0 into its line-of-sight and diffuse parts and drop the
    % line-of-sight, which the trajectory's path 0 already carries. Everything
    % is then expressed relative to that line-of-sight power.
    Klin = 10^(Kdb/10);
    losFrac = Klin/(Klin+1);
    losDb = gdb(1) + 10*log10(losFrac);          % LOS power, profile scale
    relDb = gdb - losDb;                          % clusters vs the LOS
    relDb(1) = gdb(1) + 10*log10(1-losFrac) - losDb;   % tap 0, diffuse only

    excess = tau*c;
    keep = excess <= a.MaxExcessM;
    if ~all(keep)
        warning('gen_ntn_channel:cp', ...
            '%d of %d clusters exceed the %.0f m budget and are dropped', ...
            nnz(~keep), numel(keep), a.MaxExcessM);
    end

    % ---- expand clusters into sub-rays --------------------------------------
    rng(a.Seed,'twister');
    nSub = max(1, round(a.SubRays));
    newR = []; newG = []; desc = {};
    for k = 1:numel(tau)
        if ~keep(k), continue; end
        if nSub == 1
            % One ray per cluster: no fading, a clean Doppler-shifted tap.
            % cos(theta) drawn rather than fixed at 1, because a cluster's
            % mean Doppler is not its maximum.
            cosT = 2*rand(1)-1;
        else
            % Jakes: rays uniform in arrival angle, so Doppler is fD*cos(theta)
            % and their sum fades at the difference frequencies.
            cosT = cos(2*pi*((0:nSub-1)' + rand(1))/nSub);
        end
        fd = fD*cosT;                               % Hz, per sub-ray
        rdot = -fd*c/fc;                            % m/s that implies it
        gsub = relDb(k) - 10*log10(nSub);           % split the power
        for s = 1:nSub
            % The ramp is referenced to RefTime, not to the start of the
            % table. A constant Doppler is a constant range rate, which
            % accumulates without bound - that is what a constant Doppler
            % means - so by t = 300 s a 0.83 m/s ray has drifted 249 m from
            % where it was placed. Referencing the ramp to the epoch actually
            % being transmitted puts the cluster at its nominal excess there
            % and lets it drift either side, which over a capture of seconds
            % is metres.
            newR(:,end+1) = R(:,1) + excess(k) + rdot(s)*(ts - a.RefTime); %#ok<AGROW>
            newG(:,end+1) = repmat(gsub, N, 1);                        %#ok<AGROW>
            desc{end+1} = sprintf(['%s cluster %d ray %d: excess %.1f m at ' ...
                't=%.0f s, Doppler %+.2f Hz (%.2f m/s of drift), ' ...
                '%.1f dB below the LOS'], ...
                a.Profile, k-1, s, excess(k), a.RefTime, fd(s), rdot(s), ...
                -gsub);                                                %#ok<AGROW>
        end
    end

    Rall = [R newR];
    Gall = [G newG];
    nPath = size(Rall,2);

    % ---- write ---------------------------------------------------------------
    names = cell(1, 1+2*nPath);
    names{1} = 't_s';
    for k = 1:nPath
        names{2*k}   = sprintf('R%d_m', k-1);
        names{2*k+1} = sprintf('G%d_db', k-1);
    end
    fid = fopen(fullfile(outDir,'traj.csv'),'w');
    fprintf(fid, '%s\n', strjoin(names, ','));
    fmt = ['%.6f' repmat(',%.6f,%.4f', 1, nPath) '\n'];
    body = zeros(N, 1+2*nPath);
    body(:,1) = ts;
    body(:,2:2:end) = Rall;
    body(:,3:2:end) = Gall;
    fprintf(fid, fmt, body');
    fclose(fid);

    % r_nominal must be the floor-quantised minimum over every present path, on
    % the same 1e-6 grid the CSV is written at, or a consumer that takes the
    % minimum from the file disagrees with this field by enough that fc/c
    % amplifies it into visible carrier phase.
    present = Gall > -100;
    Rp = Rall; Rp(~present) = inf;
    meta.r_nominal_m = floor(min(Rp(:))*1e6)/1e6;
    meta.carrier_hz  = fc;
    meta.ntn = struct('profile', a.Profile, 'delay_spread_s', a.DelaySpread, ...
        'mobile_speed_mps', a.MobileSpeed, 'max_doppler_hz', fD, ...
        'k_factor_first_tap_db', Kdb, 'sub_rays', nSub, ...
        'n_base_paths', nBase, 'n_added_paths', size(newR,2), ...
        'cluster_delays_s', tau(keep)', 'cluster_excess_m', excess(keep)', ...
        'cluster_rel_db', relDb(keep)', 'source', 'MATLAB nrTDLChannel');
    % meta.paths comes back from jsondecode as a struct array when every entry
    % has the same fields and as a cell array when they do not, so it is
    % normalised to a cell before anything is appended.
    base = meta.paths;
    if ~iscell(base), base = num2cell(base); end
    for k = 1:numel(desc)
        base{end+1} = struct('id', nBase+k-1, 'name', 'ntn_cluster', ...
            'description', desc{k}, 'gain_db', newG(1,k), ...
            'excess_m_max', max(newR(:,k)-Rall(:,1)), 'always_present', true); %#ok<AGROW>
    end
    meta.paths = base;
    fid = fopen(fullfile(outDir,'traj_meta.json'),'w');
    fprintf(fid,'%s\n', jsonencode(meta,'PrettyPrint',true)); fclose(fid);

    % ---- report --------------------------------------------------------------
    thr = 20.4*3/nPath;      % measured 20.4 MS/s at 3 paths, 8 interp taps
    fprintf('gen_ntn_channel -> %s\n', outDir);
    fprintf('  %s, delay spread %.0f ns, mobile %.2f m/s -> max Doppler %.2f Hz\n', ...
        a.Profile, a.DelaySpread*1e9, a.MobileSpeed, fD);
    fprintf('  K factor of tap 0 is %.1f dB, so its LOS is dropped and the\n', Kdb);
    fprintf('  trajectory''s own path 0 carries it instead\n');
    for k = 1:numel(tau)
        if ~keep(k), continue; end
        fprintf('  cluster %d: excess %7.1f m (%.0f ns), %+6.1f dB vs the LOS\n', ...
            k-1, excess(k), tau(k)*1e9, relDb(k));
    end
    fprintf('  %d base paths + %d cluster rays = %d paths\n', ...
        nBase, size(newR,2), nPath);
    % A constant Doppler is a constant range rate, so a cluster's excess path
    % drifts. Over a capture of seconds that is metres; over the whole table it
    % is the UT's own displacement, which is not a bug but does leave the CP.
    if ~isempty(newR)
        exc = newR - Rall(:,1);
        drift = max(abs(exc - (ones(N,1)*exc(abs(ts-a.RefTime)==min(abs(ts-a.RefTime)),:))), [], 1);
        fprintf('  cluster excess at t=%.0f s is nominal; over the whole %.0f s\n', ...
            a.RefTime, ts(end)-ts(1));
        fprintf('  table it drifts up to %.0f m (the UT''s own displacement).\n', max(drift));
        fprintf('  Over a 10 s capture the drift is %.1f m.\n', ...
            max(abs(rdot))*10);
        if max(exc(:)) > a.MaxExcessM
            warning('gen_ntn_channel:drift', ...
                ['cluster excess reaches %.0f m somewhere in the table, ' ...
                 'past the %.0f m budget. Transmit near RefTime, or lower ' ...
                 'MobileSpeed.'], max(exc(:)), a.MaxExcessM);
        end
    end
    fprintf('  estimated channel throughput %.1f MS/s against 7.68 required', thr);
    if thr < 7.68
        fprintf('  <-- TOO SLOW, lower SubRays\n');
    else
        fprintf('  (%.2fx)\n', thr/7.68);
    end

    out = struct('outDir', outDir, 'nPath', nPath, 'meta', meta);
end
