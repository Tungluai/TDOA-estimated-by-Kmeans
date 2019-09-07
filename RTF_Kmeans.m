function [RTF,TDOA,alphi] = RTF_Kmeans(Y,TDOA,alphi,Nspkr,frm,ref,Nbin,fs,Nfft,Nch)
% author:Xu changlai
% time : 2019/9/6
freq = 2:floor(800/fs*Nfft);
Y = (Y(:,freq).* repmat( conj(Y(ref ,freq)),Nch,1)) ./ repmat( Y(ref,freq).* conj(Y(ref ,freq)),Nch,1 );
new_alphi  = (-imag( log( Y ./ abs(Y) ))) ./  repmat((2*pi*fs*(freq-1)/Nfft),Nch,1);
if frm > 100
    alphi =  [alphi(:,size(new_alphi,2)+1:end),new_alphi] ;
else
    alphi(:,length(freq)*(frm-1)+1:length(freq)*frm ) = new_alphi;
end
if frm == 1
    %TDOA = alphi(round( rand(Nch,Nspkr)*length(freq)));   
    for i = 0: Nspkr-1
        TDOA(:,i+1) = mean(alphi,2) + (-1)^i * (i+1) * mean(alphi,2)/Nspkr; 
    end
end 
last_TDOA = TDOA - .5;
while( sum(sum((TDOA-last_TDOA).^2)) > 1e-8 )
   last_TDOA = TDOA;
   TDOA_martric = repmat(reshape(TDOA,Nch,1,Nspkr),1,size(alphi,2),1);
   alphi_martric = repmat(alphi,1,1,Nspkr);
   [~,loc_martric] = min(abs(alphi_martric-TDOA_martric),[],3);
   for pair = 1:Nch
      if pair == ref
        TDOA(pair,:) = [0,0];
        continue;
      end
      for i = 1:Nspkr
         loc = find(loc_martric(pair,:) == i);
         prob(pair,i) = length(loc);
         TDOA(pair,i) = mean(  alphi(pair,loc)  );
      end
      %对齐：假定至某时刻，各说话人的话不均等，则可由他们的话的“份量”将TDOA的列分下类，使每列的TDOA对应于某一个人
      if frm == round(0.5*fs/(Nfft/4)) % the part befor 0.5s is pre-dealing part
          [~,I] = sort(prob(pair,:),2);
          TDOA(pair,:) = TDOA(pair,I);
      end
   end
end

for bin=1:Nbin
    for sp = 1 : Nspkr
    RTF(:,sp,bin) = exp(-2*1i*pi*(bin-1)/Nfft*fs*TDOA(:,sp))/Nch;
    end
end

end

    
