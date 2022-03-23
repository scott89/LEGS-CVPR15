function [refine_lsm, lsm, propMask] = ComputeLsm(im, p)
lsm = LE(im);
os = OverSegmentation(im);

props = p.propose( os );
label = os.s();
[ PRs, propMask ] = PRscore(props, label, lsm);
propMask = double(propMask);
[~, id] = sort(PRs, 'descend');
refine_lsm = mat2gray(sum(propMask(:,:,id(1:12)), 3));

