function tlePath = make_tle(outPath, varargin)
%MAKE_TLE  Write a valid TLE for a near-circular LEO at a chosen altitude.
%
%   Synthetic, not a catalogued object: the elements are stated explicitly so
%   the pass is reproducible without a network fetch. Drop a real TLE at the
%   same path to use one instead - gen_trajectory only cares that SGP4 can
%   propagate it.

    p = inputParser;
    p.addParameter('AltKm', 550);
    p.addParameter('IncDeg', 53.0);
    p.addParameter('RAANDeg', 120.0);
    p.addParameter('Ecc', 0.0001);
    p.addParameter('ArgPDeg', 0.0);
    p.addParameter('MeanAnomDeg', 0.0);
    p.addParameter('EpochYear', 26);          % two-digit, 2026
    p.addParameter('EpochDay', 250.0);        % day-of-year, fractional
    p.addParameter('Name', 'NTN-ISAC-LEO550');
    p.parse(varargin{:});
    a = p.Results;

    mu = 398600.4418;                          % km^3/s^2, WGS84
    Re = 6378.137;
    sma = Re + a.AltKm;
    n = sqrt(mu/sma^3);                        % rad/s
    revPerDay = n*86400/(2*pi);

    l1 = sprintf('1 99001U 26001A   %02d%012.8f  .00000000  00000-0  00000-0 0  999', ...
                 a.EpochYear, a.EpochDay);
    l2 = sprintf('2 99001 %8.4f %8.4f %07.0f %8.4f %8.4f %11.8f    1', ...
                 a.IncDeg, a.RAANDeg, a.Ecc*1e7, a.ArgPDeg, a.MeanAnomDeg, revPerDay);
    l1 = [l1 sprintf('%d', tleChecksum(l1))];
    l2 = [l2 sprintf('%d', tleChecksum(l2))];
    assert(numel(l1)==69 && numel(l2)==69, 'TLE line length %d/%d', numel(l1), numel(l2));

    d = fileparts(outPath); if ~isempty(d) && ~exist(d,'dir'), mkdir(d); end
    fid = fopen(outPath,'w');
    fprintf(fid,'%s\n%s\n%s\n', a.Name, l1, l2);
    fclose(fid);
    tlePath = outPath;
    fprintf('make_tle: %s  alt=%g km  inc=%g deg  n=%.8f rev/day\n', ...
            outPath, a.AltKm, a.IncDeg, revPerDay);
end

function c = tleChecksum(line)
    c = 0;
    for i = 1:numel(line)
        ch = line(i);
        if ch >= '0' && ch <= '9'
            c = c + double(ch) - double('0');
        elseif ch == '-'
            c = c + 1;
        end
    end
    c = mod(c, 10);
end
