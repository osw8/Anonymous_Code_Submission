function [ X, Y ]=load_data(filename)











    overlap_ratio = 3/4;
    segment_size = 256;
    overlap_size = segment_size*(1-overlap_ratio);
	X = [];
	Y = [];
    load(filename);

    for i = 0 : (size(seizure_data,1)-segment_size) / overlap_size
        X = cat(3, X, seizure_data(1 + overlap_size * i:segment_size + overlap_size * i,:));
		Y = cat(1, Y, [0, 1]);
    end
    for i = 0 : (size(nonseizure_data,1)-segment_size) / segment_size
        X = cat(3, X, nonseizure_data(1 + segment_size * i:segment_size + segment_size * i,:));
		Y = cat(1, Y, [1, 0]);
    end

	rank = randperm(size(X, 3));
	X = X(:,:,rank);
	Y = Y(rank,:);
