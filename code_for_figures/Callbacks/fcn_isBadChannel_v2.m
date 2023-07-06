function y=fcn_isBadChannel_v2(eeg,ekg,Fs,LUT_BG,K)
    % eeg: 10sec 18-ch bipolar EEG
    for ii=1:size(eeg,1)
        x=eeg(ii,:);
        [eeg_nleo(ii,:),eeg_tkeo(ii,:)]=energyop(x,0);
    end
    [ekg_nleo,ekg_tkeo]=energyop(ekg,0);
  
    ff1=fcn_computeFeatures_powers_complex_v2(eeg,ekg,Fs);
    ff2=fcn_computeFeatures_powers_complex_v2(eeg_nleo,ekg_nleo,Fs);
    ff3=fcn_computeFeatures_powers_complex_v2(eeg_tkeo,ekg_tkeo,Fs);
    
    ff=[ff1,ff2,ff3];
    y=fcn_bcReject(ff,LUT_BG,K);
end