% LCMV-GSC for speech enhancement
% author : Xu Changlai,6/2,2019
warning('off');
clear all
close all

[speech , fs ] = audioread('male_female_pure_mixture.wav');
speech = speech';
speech = [speech,speech];
[Nch,Nz] = size(speech);
Nfft =floor( fs*128/1000); % 64 ms per frame
Nshift = floor(Nfft/4);
Nbin = Nfft/2 +1;
Nfrm = floor(Nz/Nshift)-3;
win = sqrt(hanning(Nfft))';

yout = zeros(1,Nz);
Ybin_nonclosed = zeros(1,Nbin);
 
q = zeros(Nch+1, Nbin);
pest = zeros(Nbin,1);
mu = 0.05;
alphaP = 0.9;
Yfbf = zeros(Nch,Nfft);
phi_x = zeros(Nbin);
phi_n = zeros(Nbin);
PhiN = zeros(Nch+1, Nch+1, Nbin);
PhiS = zeros(Nch+1, Nch+1, Nbin);
PhiSN = zeros(Nch+1,Nbin);


nsrce = 2;
freq = 2:floor(800/fs*Nfft);
TDOA = zeros(Nch,nsrce);
alphi = zeros(Nch,length(freq));
g = [1;zeros(nsrce-1,1)];
g = flipud(g); % change the enhanced person in the case of 2 speakers
enhansp = find(1 == g); 
ref = ceil(rand(1)*Nch);% randomly select a channel as reference channel
disp(ref);
%%
 for frm = 1 : Nfrm  
    freqaxis = (frm-1)*Nshift+1:(frm-1)*Nshift+Nfft;
    %STFT
    for ch = 1 : Nch
         Y(ch ,:) = fft(win .* speech(ch ,freqaxis),Nfft);
    end
    [C,TDOA,alphi] = RTF_Kmeans(Y,TDOA,alphi,nsrce,frm,ref,Nbin,fs,Nfft,Nch);
    for bin=1:Nbin  
        base(:,:,bin) = (C(:,:,bin)'* C(:,:,bin));
        %check the martrix
        if rcond(base(:,:,bin)) < eps
            base(:,:,bin) = base(:,:,bin) + 1e-10 * eye(size(base(:,:,bin),1));
        end
        w0(:,bin) = C(:,:,bin)/base(:,:,bin) * g;   
        B(:,:,bin) = eye(Nch,Nch) - C(:,:,bin) /base(:,:,bin)*C(:,:,bin)';

    % processing      
        % FBF filtering
        Yfbf(bin) = w0(:,bin)' * Y(:,bin)/norm(w0(:,bin));    
        % BM filtering
        u(:,bin) = B(:,:,bin) * Y(:,bin);
        
        Yout(bin) = Yfbf(bin) - q(:,bin)'* [Yfbf(bin);u(:,bin)];

        % SDW-MWF
        S(:,bin) = [Yout(bin);1e-10 * ones(Nch,1)];
        N(:,bin) = [Yfbf(bin)-Yout(bin); u(:,bin)];
        PhiS(:,:,bin) = 0.98 * PhiS(:,:,bin) + 0.02 * S(:,bin) * S(:,bin)';
        PhiN(:,:,bin) = 0.98 * PhiN(:,:,bin) + 0.02 * N(:,bin) * N(:,bin)';
        PhiSN(:,bin) = 0.98 * PhiSN(:,bin) + 0.02 * N(:,bin) * (Yfbf(bin)-Yout(bin))';
        q(:,bin) = MWF(PhiS(:,:,bin), PhiN(:,:,bin),PhiSN(:,bin),1.11); 
    end
    % load all stuff
    yout(freqaxis) = yout(freqaxis) + win .* real(ifft([Yout,conj(Yout(end-1:-1:2))]));   
end
audiowrite('RTF.wav', yout(end-Nz/2:end),fs);
%%
% plot
figure(2);

subplot(4,1,1);
plot(audioread('male.wav'));

subplot(4,1,2);
plot(audioread('female.wav'));

subplot(4,1,3);
plot(speech(1,:));

subplot(4,1,4);
plot(audioread('RTF.wav'));

[ scoresbefore ] = pesq( 'male.wav', 'male_female_pure_mixture.wav' );
[ scoresafter ] = pesq( 'male.wav', 'RTF.wav' );
[ scoresideal ] = pesq( 'male.wav', 'male.wav' );
fprintf('scorebefore: %f\n',scoresbefore);
fprintf('scoreafter: %f\n',scoresafter);
fprintf('scoreideal: %f\n',scoresideal);
fprintf(['improved PESQ socre : %f\n'],scoresafter-scoresbefore);

[ scoresbefore ] = pesq( 'female.wav', 'male_female_pure_mixture.wav' );
[ scoresafter ] = pesq( 'female.wav', 'RTF.wav' );
[ scoresideal ] = pesq( 'female.wav', 'female.wav' );
fprintf('scorebefore: %f\n',scoresbefore);
fprintf('scoreafter: %f\n',scoresafter);
fprintf('scoreideal: %f\n',scoresideal);
fprintf(['improved PESQ socre : %f\n'],scoresafter-scoresbefore);