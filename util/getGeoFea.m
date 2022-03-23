function [geoFeature] = getGeoFea(objMask)
% Compute the geometic property of each object proposal
% INPUT:
%               - ojbMask: the proposal mask, imHeight x imWidth x objPropNum
% OUTPUT:
%              - geoFeature: the geometric feature of each object
%              proposal, objPropNum x 8, including: 12) the normalized centroid
%              coordinates, 34) the normalized major and minor axis length, 5) Euler number, 678) the normalized
%              width and height and aspect ratio of the bounding box.
properties = {'BoundingBox', 'Centroid', 'MajorAxisLength' ,'MinorAxisLength',  'EulerNumber'};
[h, w, propNum] = size(objMask);
geoFeature = zeros(propNum, 9);
parfor pId = 1:propNum
    fea = struct2array(regionprops(objMask(:,:, pId), properties));
%     try
    geoFeature(pId, :) = fea;
%     catch aa;
%         aa=1;
%     end
end
geoFeature = geoFeature(:, [3,4,5,6, 1,2, 7, 8, 9 ]);
geoFeature = [geoFeature(:, 3)./geoFeature(:,4) geoFeature(:, 3:end)];  % kick out the topleft coordinates and add in the aspectratio
%% geoFeature: aspectratio boxW boxH cx cy majorl minorl Eulernum
%%                            1             2        3     4   5     6          7           8
geoFeature(:, [2, 4]) = geoFeature(:, [2, 4])/w;
geoFeature(:, [3, 5]) = geoFeature(:, [3, 5])/h;
geoFeature(:, [6, 7]) = geoFeature(:, [6, 7])/(w+h);

return


