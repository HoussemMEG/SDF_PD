clear
clc

% /!\ On the first run it is mandatory to slect "change path" instead of
% "add to path", or you should be in the working directory so that "./" is
% known and in the right place

% -----------------------------------------------------------------------
% This file removes the ICs that contain ocular activity and crop the
% epochs yielding epochs from starting from 0 to +500 ms post stimulus
%
% Adapted from computational tool t001: http://predict.cs.unm.edu/tools.php
% Step1_PreProcess_ODDBALL.m
% -----------------------------------------------------------------------


% Add EEGLAB function to path
addpath(genpath('./EEGLAB'));

% Set save directory
save_dir_name = '1-30[Hz] + EB ICA removed/';
save_dir = strcat('./data/processed/', save_dir_name);
mkdir(save_dir)  % create the directory

% Parkinsons Disease and Control file number
PD = [804 805 806 807 808 809 810 811 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829];
CTL = [890 891 892 893 895 896 897 898 899 900 901 902 903 904 905 906 907 909 910 911 912 913 914 8060 8070];

eeg_chans =1:60;

for subj=[PD, CTL]
    for session=1:2
        if (subj>850 && session==1) || subj<850  % If not ctl, do session 2
            if (exist(['data/', num2str(subj),'_',num2str(session),'_PDDys_ODDBALL.mat']))
                disp(['Oddball ERPs --- Subno: ',num2str(subj),'      Session: ',num2str(session)]);
                
                load(['data/', num2str(subj),'_',num2str(session),'_PDDys_ODDBALL.mat'],'EEG','bad_chans','bad_epochs','bad_ICAs');
                disp('Loading done')
                
                % Eye blink ICA removal
                bad_ICAs_To_Remove=1;         % set '1' as default - omit based on visual selection below
                if session==1
                    if any(subj==[817 818 825 890 895 897 902 910 913]), bad_ICAs_To_Remove=[1,2];
                    elseif any(subj==[819 823 826 828 891 894 896 900 906 907 909]), bad_ICAs_To_Remove=[1,2,3];
                    elseif subj==829, bad_ICAs_To_Remove=2;
                    elseif subj==905, bad_ICAs_To_Remove=3;
                    elseif subj==805, bad_ICAs_To_Remove=5;
                    elseif subj==911, bad_ICAs_To_Remove=5;
                    elseif subj==820, bad_ICAs_To_Remove=6;
                    elseif subj==827, bad_ICAs_To_Remove=8;
                    elseif subj==808, bad_ICAs_To_Remove=[];
                    % elseif subj==824, bad_ICAs_To_Remove=[1, 2, 5, 7, 10, 11, 12, 13, 14, ];
                    end
                elseif session==2
                    if any(subj==[811 814 818 819 826]), bad_ICAs_To_Remove=[1,2];
                    elseif subj==817, bad_ICAs_To_Remove=[1,2,3];
                    elseif subj==823, bad_ICAs_To_Remove=[1,2,3,4];
                    elseif subj==825, bad_ICAs_To_Remove=[1,2,3,4,5];
                    elseif subj==805, bad_ICAs_To_Remove=18;
                    elseif subj==806, bad_ICAs_To_Remove=5;
                    elseif subj==829, bad_ICAs_To_Remove=4;
                    elseif subj==815, bad_ICAs_To_Remove=3;
                    end
                end
                
                if ~isempty(bad_ICAs_To_Remove)
                    EEG = pop_subcomp( EEG, bad_ICAs_To_Remove, 0);
                    disp('ICA removed')
                end
                
                
                %                 % FASTER
                %                 epoch = epoch_properties(EEG,eeg_chans);
                %                 epoch_exceeded_threshold = min_z_JFC(epoch);  % Cols are: 1) mean epoch deviation, 2) epoch variance, 3) max amplitude
                %                 FASTER_bad_epochs = logical(epoch_exceeded_threshold(:,1)+epoch_exceeded_threshold(:,2)+epoch_exceeded_threshold(:,3)); % ANYTHING marked as bad is bad
                %
                %                 % REJECT
                %                 binarized=zeros(1,EEG.trials);
                %                 binarized(FASTER_bad_epochs)=1;    % Only the FASTER ones
                %                 EEG = pop_rejepoch(EEG,binarized,0);
                % -------------- Downsample
                %                 EEG=pop_resample(EEG,100);
                %                 EEG.srate = 100;
                
                % Filter
                dims=size(EEG.data);
                EEG.data=eegfiltfft(EEG.data,EEG.srate,1, 30);
                EEG.data=reshape(EEG.data,dims(1),dims(2),dims(3));
                
                % Get the right stimulus onset
                dims=size(EEG.data);
                t_min = 0; % We want t_min before stimulus onset % keep value 0
                t_max = 498; % t_max after stimulus onset % keep value 498
                EEG.xmin = t_min;
                EEG.xmax = t_max;
                tx = -2000:(1000/EEG.srate):1998; % Creating time vector (care from the downsampling)
                
                
                % Find index of t_min and t_max for all stimuli, 
                % the shift is due to some loading latencies encountred 
                % during the experiment
                N_t1 = find(tx==t_min);  N_t2 = find(tx==t_max); % indexes for Novel stimuli
                ST_t1 = find(tx==t_min+450);  ST_t2 = find(tx==t_max+450); % indexes for Std & Target stimuli
                
                
                % Get stimuli type and store it in EEG.epoch.STIM
                for ai=1:size(EEG.epoch,2) % For each epoch
                    for bi=1:size(EEG.epoch(ai).eventlatency,2) % For each stimulus in the epoch window
                        if EEG.epoch(ai).eventlatency{bi}==0 % Get the right stimuli  of that epoch
                            FullName = EEG.epoch(ai).eventtype{bi};
                            EEG.epoch(ai).STIM = str2num(FullName(2:end)) ;
                        end
                    end
                end
                
           
                % Creating new variable
                EEG_temp = EEG;
                EEG.data = [];
                EEG.VEOG = [];
                for ai=1:size(EEG_temp.epoch,2)
                    EEG.epoch(ai).STIM = EEG_temp.epoch(ai).STIM;
                    if EEG.epoch(ai).STIM==202 %if novel
                        EEG.data(:,:,ai) = EEG_temp.data(:,N_t1:N_t2,ai);
                        EEG.VEOG(:,ai) = EEG_temp.VEOG(N_t1:N_t2, ai);
                    else
                        EEG.data(:,:,ai) = EEG_temp.data(:,ST_t1:ST_t2,ai);
                        EEG.VEOG(:,ai) = EEG_temp.VEOG(ST_t1:ST_t2, ai);
                    end
                end
                
                EEG.times = EEG.times(N_t1:N_t2);
                EEG.pnts = length(EEG.times);
                save([save_dir, num2str(subj),'_',num2str(session),'_PDDys_ODDBALL.mat'], 'EEG', 'bad_epochs')
                disp('Saving done'); disp(' ')
                clear EEG_temp EEG
            end
        end
    end
end



