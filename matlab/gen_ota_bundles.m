function gen_ota_bundles(varargin)
%GEN_OTA_BUNDLES  Waveform and trajectory bundles for the over-the-air runs.
%
%   Regenerates everything the two-laptop OTA setup needs, at the 2.4 GHz ISM
%   carrier rather than the 2.0 GHz of the conducted work.
%
%   Why 2478 MHz. The occupied bandwidth is 4.5 MHz and the sample rate is
%   7.68 MS/s, so about 7.7 MHz has to be clear. Wi-Fi channels 1, 6 and 11
%   occupy 2401-2423, 2426-2448 and 2451-2473, leaving 3 MHz gaps between
%   them - too narrow - and a 10 MHz gap above channel 11. Centring at
%   2478 MHz puts the signal in 2474.2-2481.8 MHz, clear of 1/6/11 and
%   overlapping only channel 13, which is not used in the US, and BLE
%   advertising channel 39 at 2480 MHz. Run tools/spectrum_survey.py first;
%   if the local band looks different, override with 'CarrierHz'.
%
%   Under FCC part 15.247 this is a digital modulation system - the 6 dB
%   bandwidth is far above the 500 kHz floor - so it is licence-free up to
%   30 dBm conducted with a 6 dBi antenna. A B200 delivers about 10 dBm.
%
%   Bundles written:
%     data/wf_ota      100% PRB occupancy. The default for OTA: findings.md
%                      section 4 shows a thinned aperture has a sidelobe
%                      pedestal near -10log10(N_active) dBc, and a real
%                      target sits at delay bin 0 under the direct path,
%                      which is exactly where that pedestal is worst.
%     data/wf_ota050   50% random, for the occupancy comparison.
%     data/traj_ota    the LEO pass, with carrier-scaled diagnostics.

    p = inputParser;
    p.addParameter('CarrierHz', 2.478e9);
    p.addParameter('Root', fullfile(fileparts(mfilename('fullpath')), '..'));
    p.parse(varargin{:});
    a = p.Results;

    addpath(fileparts(mfilename('fullpath')));
    data = fullfile(a.Root, 'data');

    fprintf('=== OTA bundles at %.3f GHz ===\n', a.CarrierHz/1e9);

    gen_waveform('OutDir', fullfile(data,'wf_ota'), ...
                 'Occupancy', 1.0, 'CarrierHz', a.CarrierHz);
    gen_waveform('OutDir', fullfile(data,'wf_ota050'), ...
                 'Occupancy', 0.5, 'PRBPattern', 'random', 'CarrierHz', a.CarrierHz);
    gen_trajectory('OutDir', fullfile(data,'traj_ota'), 'CarrierHz', a.CarrierHz);

    fprintf('\nDone. Both laptops need data/wf_ota and data/traj_ota.\n');
end
