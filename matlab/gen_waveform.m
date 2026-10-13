function out = gen_waveform(varargin)
%GEN_WAVEFORM  NR downlink waveform generator for the NTN ISAC prototype.
%
%   Emits the file-interface bundle described in handoff section 3 (plus the
%   two additions documented in docs/file-interface.md) into an output folder:
%
%     tx_iq.cf32    interleaved float32 LE complex baseband
%     tx_grid.c64   interleaved float32 LE grid, C order [n_sym_total, n_sc]
%     ref_mask.u8   1 byte/RE, 1 where the RE is a regenerable reference
%     occ_mask.u8   1 byte/RE, 1 where the RE carries any transmitted energy
%     ref_syms.c64  sparse: values of the masked reference REs, raster order
%     config.json   parameters + file manifest
%
%   Name-value arguments (all optional):
%     OutDir       output folder                        (default ../data/wf)
%     Occupancy    PDSCH PRB occupancy fraction 0<f<=1  (default 0.5)
%     Seed         RNG seed for the PRB pattern         (default 1234)
%     Modulation   'QPSK' | '16QAM' | '64QAM'           (default 'QPSK')
%     TargetRate   PDSCH target code rate               (default 308/1024, MCS 4)
%     NCellID      physical cell id                     (default 1)
%     NSlots       slots in the file                    (default 20)
%     RmsTarget    RMS of tx_iq after scaling           (default 0.1)
%     CarrierHz    RF carrier recorded in config.json   (default 2.0e9)
%
%   CarrierHz does not change a single sample of tx_iq - the waveform is
%   complex baseband and the carrier only enters downstream, in the channel
%   block's phase term and in the receiver's Doppler budget. It is recorded
%   here so that one bundle fully determines the link. For over-the-air runs
%   in the 2.4 GHz ISM band, pass CarrierHz 2.478e9; see docs/ota.md for why
%   that particular channel, and note that the higher carrier scales the pass
%   Doppler by fc/2.0e9, which widens the acquisition search.
%
%   The grid is written column-major from the MATLAB [n_sc x n_sym] array,
%   which is byte-identical to C order [n_sym, n_sc]: index = l*n_sc + k.

    p = inputParser;
    p.addParameter('OutDir', fullfile(fileparts(mfilename('fullpath')),'..','data','wf'));
    p.addParameter('Occupancy', 0.5, @(x) isscalar(x) && x > 0 && x <= 1);
    p.addParameter('Seed', 1234);
    p.addParameter('PRBPattern', 'random');    % 'random' | 'contiguous'
    p.addParameter('Modulation', 'QPSK');
    p.addParameter('TargetRate', 308/1024);
    p.addParameter('NCellID', 1);
    p.addParameter('NSlots', 20);
    p.addParameter('RmsTarget', 0.1);
    p.addParameter('CarrierHz', 2.0e9);
    p.parse(varargin{:});
    a = p.Results;

    nPRB = 25; nSC = nPRB*12; scsKHz = 15; nSlots = a.NSlots;
    nSymSlot = 14; nSymTotal = nSlots*nSymSlot;

    % ---- carrier / bandwidth part -------------------------------------------
    cfg = nrDLCarrierConfig;
    cfg.NCellID           = a.NCellID;
    cfg.NumSubframes      = nSlots;          % 15 kHz => 1 slot per subframe
    cfg.WindowingPercent  = 0;               % required: keeps tx_grid exact
    cfg.FrequencyRange    = 'FR1';
    cfg.ChannelBandwidth  = 5;
    cfg.CarrierFrequency  = 0;               % no phase compensation
    cfg.SCSCarriers   = {nrSCSCarrierConfig('SubcarrierSpacing',scsKHz, ...
                            'NSizeGrid',nPRB,'NStartGrid',0)};
    cfg.BandwidthParts = {nrWavegenBWPConfig('BandwidthPartID',1, ...
                            'SubcarrierSpacing',scsKHz,'NSizeBWP',nPRB,'NStartBWP',0)};
    cfg.CORESET = {}; cfg.SearchSpaces = {}; cfg.PDCCH = {}; cfg.CSIRS = {};

    % ---- SSB: one block, slot 0 symbols 2..5, period = file length ----------
    ncrbSSB = floor((nPRB - 20)/2);          % centre the 20-PRB SSB in 25 PRB
    cfg.SSBurst = nrWavegenSSBurstConfig( ...
        'BlockPattern','Case A','TransmittedBlocks',[1 0 0 0], ...
        'Period',nSlots,'NCRBSSB',ncrbSSB,'KSSB',0, ...
        'DMRSTypeAPosition',2,'Power',0);

    % ---- PDSCH: one config per slot so PRB occupancy can vary per slot ------
    % Allocation pattern is not a cosmetic choice. The delay axis of the
    % sensing map is the Fourier transform of the occupied subcarriers, so a
    % scattered PRB set is a thinned aperture: full delay resolution, but a
    % sidelobe pedestal near -10*log10(N_active) dBc that masks weak echoes
    % everywhere. A contiguous set is a filled but narrower aperture: coarser
    % delay resolution, low sidelobes. Both are realistic schedulers.
    rng(a.Seed,'twister');
    nAlloc = max(1, round(a.Occupancy*nPRB));
    prbSets = cell(1,nSlots-1);
    pdschCfgs = cell(1,nSlots-1);
    for s = 1:nSlots-1                        % slot 0 is SSB only
        switch lower(a.PRBPattern)
            case 'random'
                prb = sort(randperm(nPRB, nAlloc)) - 1;
            case 'contiguous'
                st = randi(nPRB - nAlloc + 1) - 1;
                prb = st + (0:nAlloc-1);
            otherwise
                error('unknown PRBPattern %s', a.PRBPattern);
        end
        prbSets{s} = prb;
        pd = nrWavegenPDSCHConfig( ...
            'BandwidthPartID',1,'Modulation',a.Modulation,'NumLayers',1, ...
            'MappingType','A','SymbolAllocation',[1 13],'PRBSet',prb, ...
            'RNTI',1,'NID',a.NCellID,'Coding',true,'TargetCodeRate',a.TargetRate, ...
            'Power',0,'DMRSPower',0,'SlotAllocation',s,'Period',nSlots, ...
            'DataSource','PN9','Label',sprintf('PDSCH slot %d',s));
        pd.DMRS = nrPDSCHDMRSConfig('DMRSConfigurationType',1, ...
            'DMRSTypeAPosition',2,'DMRSAdditionalPosition',1,'DMRSLength',1, ...
            'NumCDMGroupsWithoutData',1);
        pdschCfgs{s} = pd;
    end
    cfg.PDSCH = pdschCfgs;

    [wv, info] = nrWaveformGenerator(cfg);
    assert(numel(wv) == nSlots*7680, 'unexpected waveform length %d', numel(wv));

    % ---- the transmitted grid, recovered exactly from the waveform ----------
    % nrWaveformGenerator sums the SSB in the time domain and does not merge it
    % into ResourceGridInCarrier, so demodulating is the only way to get a grid
    % that describes the whole signal. Verified round-trip exact to ~3e-16.
    car = nrCarrierConfig('NSizeGrid',nPRB,'SubcarrierSpacing',scsKHz, ...
                          'NSlot',0,'NFrame',0);
    grid = nrOFDMDemodulate(car, wv, 'CyclicPrefixFraction', 0);
    assert(isequal(size(grid), [nSC nSymTotal]), 'grid size %s', mat2str(size(grid)));

    chk = nrOFDMModulate(car, grid, 'Windowing', 0);
    relErr = norm(chk - wv)/norm(wv);
    assert(relErr < 1e-10, 'grid/waveform round-trip broken: %.3e', relErr);

    % ---- masks ---------------------------------------------------------------
    refMask = false(nSC, nSymTotal);

    % PDSCH DMRS
    for k = 1:numel(info.WaveformResources.PDSCH)
        for r = info.WaveformResources.PDSCH(k).Resources(:)'
            [sc, sym] = ind2sub([nSC nSymSlot], double(r.DMRSIndices));
            refMask(sub2ind([nSC nSymTotal], sc, sym + double(r.NSlot)*nSymSlot)) = true;
        end
    end

    % SSB: only the parts a receiver can regenerate from NCellID.
    % PBCH payload is deliberately excluded from refMask - it is not known
    % without decoding the MIB, so calling it a reference would flatter the
    % reference-only branch. It is still occupied, so it goes in occMask.
    ssbRef = false(240,4); ssbOcc = false(240,4);
    ssbRef(nrPSSIndices())                = true;
    ssbRef(nrSSSIndices())                = true;
    ssbRef(nrPBCHDMRSIndices(a.NCellID))  = true;
    ssbOcc(nrPBCHIndices(a.NCellID))      = true;
    ssbOcc = ssbOcc | ssbRef;
    k0 = ncrbSSB*12; l0 = 2;                  % SSB origin in the carrier grid
    ssbRows = k0+(1:240); ssbCols = l0+(1:4);
    refMask(ssbRows, ssbCols) = refMask(ssbRows, ssbCols) | ssbRef;

    % occMask is built structurally, not by thresholding the grid: demodulation
    % leaves ~1e-17 of numerical dust on unallocated REs and a threshold test
    % silently reports 100%% occupancy. This is also the form an OCUDU grid hook
    % would emit.
    occMask = false(nSC, nSymTotal);
    occMask(ssbRows, ssbCols) = occMask(ssbRows, ssbCols) | ssbOcc;
    for k = 1:numel(info.WaveformResources.PDSCH)
        for r = info.WaveformResources.PDSCH(k).Resources(:)'
            idx = double([r.ChannelIndices(:); r.DMRSIndices(:)]);
            [sc, sym] = ind2sub([nSC nSymSlot], idx);
            occMask(sub2ind([nSC nSymTotal], sc, sym + double(r.NSlot)*nSymSlot)) = true;
        end
    end
    assert(all(occMask(refMask)), 'ref_mask marks REs with no transmitted energy');

    % Cross-check the structural mask against energy, then scrub the dust so
    % tx_grid.c64 is exactly zero off-allocation.
    reRms = sqrt(mean(abs(grid(occMask)).^2));
    leak  = max(abs(grid(~occMask)));
    assert(leak < 1e-8*reRms, 'unallocated REs carry energy: %.3e vs %.3e', leak, reRms);
    assert(min(abs(grid(occMask))) > 1e-6*reRms, 'occ_mask marks an empty RE');
    grid(~occMask) = 0;
    relErr = norm(nrOFDMModulate(car, grid, 'Windowing', 0) - wv)/norm(wv);
    assert(relErr < 1e-10, 'round-trip broken after dust scrub: %.3e', relErr);

    % ---- scale ---------------------------------------------------------------
    alpha = a.RmsTarget / rms(wv);
    wv = wv*alpha; grid = grid*alpha;
    pk = max(abs(wv));
    assert(pk < 0.9, 'peak %.3f exceeds 0.9 - reduce RmsTarget', pk);

    refSyms = grid(refMask);                  % column-major == raster C order

    % ---- write ---------------------------------------------------------------
    od = a.OutDir;
    if ~exist(od,'dir'), mkdir(od); end
    writecf32(fullfile(od,'tx_iq.cf32'),  wv);
    writecf32(fullfile(od,'tx_grid.c64'), grid);
    writecf32(fullfile(od,'ref_syms.c64'),refSyms);
    writeu8(  fullfile(od,'ref_mask.u8'), refMask);
    writeu8(  fullfile(od,'occ_mask.u8'), occMask);

    oi = nrOFDMInfo(car);
    cj = struct();
    cj.generator      = 'matlab';
    cj.generated_utc  = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
    cj.fs_hz          = oi.SampleRate;
    cj.carrier_hz     = a.CarrierHz;
    cj.scs_khz        = scsKHz;
    cj.n_prb          = nPRB;
    cj.n_sc           = nSC;
    cj.n_fft          = oi.Nfft;
    cj.cp_lengths     = oi.CyclicPrefixLengths;
    cj.cp_fraction    = 0;
    cj.windowing      = 0;
    cj.n_slots        = nSlots;
    cj.n_sym_slot     = nSymSlot;
    cj.n_sym_total    = nSymTotal;
    cj.grid_shape     = [nSymTotal nSC];
    cj.frame_len_samples = numel(wv);
    cj.loopable       = true;
    cj.ncellid        = a.NCellID;
    cj.tx_scale       = alpha;
    cj.tx_rms         = rms(wv);
    cj.tx_peak        = pk;
    cj.papr_db        = 20*log10(pk/rms(wv));
    cj.n_ref_re       = nnz(refMask);
    cj.n_occ_re       = nnz(occMask);
    cj.dmrs           = struct('config_type',1,'symbols',[2 11], ...
                               'type_a_position',2,'additional_position',1, ...
                               'cdm_groups_without_data',1);
    cj.pdsch          = struct('modulation',a.Modulation,'target_code_rate',a.TargetRate, ...
                               'start_symbol',1,'n_symbols',13,'mapping_type','A', ...
                               'rnti',1,'slots',1:nSlots-1);
    cj.ssb            = struct('block_pattern','Case A','ncrb_ssb',ncrbSSB,'kssb',0, ...
                               'slot',0,'symbols',[2 3 4 5],'n_sc',240);
    cj.occupancy      = struct('fraction',a.Occupancy,'seed',a.Seed, ...
                               'pattern',lower(a.PRBPattern), ...
                               'n_prb_alloc',nAlloc,'prb_sets',{prbSets});
    cj.files          = manifest(od, {'tx_iq.cf32','tx_grid.c64','ref_syms.c64', ...
                                      'ref_mask.u8','occ_mask.u8'});
    fid = fopen(fullfile(od,'config.json'),'w');
    fprintf(fid,'%s\n', jsonencode(cj,'PrettyPrint',true)); fclose(fid);

    fprintf('gen_waveform -> %s\n', od);
    fprintf('  %d samples, %.1f ms, rms %.4f, peak %.4f, PAPR %.2f dB\n', ...
        numel(wv), 1e3*numel(wv)/oi.SampleRate, rms(wv), pk, cj.papr_db);
    fprintf('  grid %dx%d, ref REs %d (%.2f%%), occupied REs %d (%.2f%%)\n', ...
        nSymTotal, nSC, cj.n_ref_re, 100*cj.n_ref_re/(nSC*nSymTotal), ...
        cj.n_occ_re, 100*cj.n_occ_re/(nSC*nSymTotal));
    fprintf('  round-trip grid<->iq rel-err %.2e\n', relErr);

    out = struct('outDir',od,'waveform',wv,'grid',grid,'refMask',refMask, ...
                 'occMask',occMask,'config',cj);
end

% -------------------------------------------------------------------------
function writecf32(path, x)
    x = x(:);
    v = zeros(2*numel(x),1,'single');
    v(1:2:end) = single(real(x));
    v(2:2:end) = single(imag(x));
    fid = fopen(path,'w','ieee-le');
    assert(fid > 0, 'cannot open %s', path);
    fwrite(fid, v, 'float32'); fclose(fid);
end

function writeu8(path, m)
    fid = fopen(path,'w','ieee-le');
    assert(fid > 0, 'cannot open %s', path);
    fwrite(fid, uint8(m(:)), 'uint8'); fclose(fid);
end

function s = manifest(od, names)
    s = struct();
    for i = 1:numel(names)
        d = dir(fullfile(od,names{i}));
        key = matlab.lang.makeValidName(names{i});
        s.(key) = struct('name',names{i},'bytes',d.bytes);
    end
end
