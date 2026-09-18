function [ X_1, X_2, X_3]=domain_transform(X)














    L = size(X, 1);
    chan = size(X, 2);
    num = size(X, 3);

    X_1 = [];
    X_2 = [];
    X_3 = [];

    fprintf('    transform time feature\n');
    for i = 1:num
        a = [];
        for j = 1:chan
            a = cat(1, a, X(:,j,i));
        end
        X_1 = cat(1, X_1, a'); 
    end

    fprintf('    transform frequency feature\n');
    NFFT = 2 ^ nextpow2(L);
    for i = 1:num
        b = [];
        for j = 1:chan
            a = fft(X(:,j,i),NFFT) / L;
            a = 2 * abs(a);
            b = cat(1, b, a(4:30));
        end
        X_2 = cat(1, X_2, b');
    end

    fprintf('    transform time-frequency feature\n');
    WPD_layers = 6;
    wavelet_basis = 'db4';
    fs = 256;
    for i = 1:num
        a = [];
        for j = 1:chan
            wpt = wpdec(X(:,j,i), WPD_layers, wavelet_basis);
            [SPEC,~,~] = wpspectrum(wpt,fs);
            a = cat(2, a, reshape(SPEC(2:15,:)', [14 * 256,1])');
        end
        X_3 = cat(1, X_3, a);                
    end
end
