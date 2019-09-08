function [RTF,TDOA,alphi,real_loc] = RTF_Kmeans(Y,TDOA,alphi,real_loc,Nspkr,frm,ref,Nbin,fs,Nfft,Nch)
% author:Xu changlai
% time : 2019/9/6
x = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9];
freq = 2:floor(700/fs*Nfft); % Roughly estimate that 0 ~ 700Hz is non-overlapped
Y = (Y(:,freq).* repmat( conj(Y(ref ,freq)),Nch,1)) ./ repmat( Y(ref,freq).* conj(Y(ref ,freq)),Nch,1 );
new_alphi  = (-imag( log( Y ./ abs(Y) ))) ./  repmat((2*pi*fs*(freq-1)/Nfft),Nch,1);
if frm > 100
    alphi =  [alphi(:,size(new_alphi,2)+1:end),new_alphi] ;
else
    alphi(:,length(freq)*(frm-1)+1:length(freq)*frm ) = new_alphi;
end
% initializing
if frm == 1  
    for i = 0: Nspkr-1
        TDOA(:,i+1) = mean(alphi,2) + (-1)^i * x(i+1) * mean(alphi,2)/Nspkr; 
    end
end 

% Kmeans
last_TDOA = TDOA - .5;
while( sum(sum((TDOA-last_TDOA).^2)) > 1e-8 )
   last_TDOA = TDOA;
   TDOA_martric = repmat(reshape(TDOA,Nch,1,Nspkr),1,size(alphi,2),1);
   alphi_martric = repmat(alphi,1,1,Nspkr);
   [~,loc_martric] = min(abs(alphi_martric-TDOA_martric),[],3);
   for pair = 1:Nch
      if pair == ref
        prob(pair,:) = zeros(1,Nspkr);
        continue;
      end
      for i = 1:Nspkr
         loc = find(loc_martric(pair,:) == i);
         prob(pair,i) = length(loc);
         TDOA(pair,i) = mean(  alphi(pair,loc)  );
      end
      % to make sure one column of TDOA belong to only one person(classified by the difference of speakers' speech propotion )
      if frm < 10
         [~,I] = sort(prob(pair,:),2);
         TDOA(pair,:) = TDOA(pair,I);
      end
   end  
end

% If there is a channel performed badly, then revise it 
detaN = prob(:,1)- prob(:,2);
[~,loc_max] = max(detaN);     
[~,loc_min] = min(detaN);
detaN2 = detaN;
prob2 = prob;
if ref == real_loc
   detaN2(ref)=[];
   prob2(real_loc,:)=[];
else
   detaN2(max(ref,real_loc))=[];
   detaN2(min(ref,real_loc))=[];
   prob2(max(ref,real_loc),:)=[];
   prob2(min(ref,real_loc),:)=[];
end
mean_detaN = sum(detaN2)/(length(detaN2));
if loc_max == ref
    real_loc = loc_min;
elseif loc_min == ref
    real_loc = loc_max;
elseif abs(detaN(loc_max)-mean_detaN) > abs(detaN(loc_min)-mean_detaN)
    real_loc = loc_max;
else 
    real_loc = loc_min;
end
if abs(detaN(real_loc)-mean_detaN)> min([abs(mean_detaN),100])
   for i = 0: Nspkr-1
       TDOA(real_loc,i+1) =  mean(alphi(real_loc,:)) + (-1)^i *x(i+1) * mean(alphi(real_loc,:))/Nspkr;
   end
   last_ral_TDOA = TDOA(real_loc,:) - .5;
   while(sum((TDOA(real_loc,:)-last_ral_TDOA).^2)>1e-8)
      last_ral_TDOA  = TDOA(real_loc,:);
      TDOA_real_martric = repmat(reshape(TDOA(real_loc,:),1,1,Nspkr),1,size(alphi,2),1);
      alphi_real_martric = repmat(alphi(real_loc,:),1,1,Nspkr);
      [~,loc_real_martric] = min(abs(alphi_real_martric-TDOA_real_martric),[],3);
      for i = 1:Nspkr
         loc1 = find(loc_real_martric == i);
         prob1(i) = length(loc1);
         TDOA(real_loc,i) = mean(  alphi(real_loc,loc1)  );
      end  
   end  
   [~,I] = sort(prob1);
   TDOA(real_loc,:) = TDOA(real_loc,I);
   [~,I_now] = sort(sum(prob2,1));
   TDOA(real_loc,:) = TDOA(real_loc,I_now);
end
   
   

% Use the estimated TDOA estimate RTF
for bin=1:Nbin
    for sp = 1 : Nspkr
    RTF(:,sp,bin) = exp(-2*1i*pi*(bin-1)/Nfft*fs*TDOA(:,sp))/Nch;
    end
end

end

    
