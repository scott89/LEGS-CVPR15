function objMask = getObjMask(props, label)

propNum = size(props, 1);
imsz = size(label);

objMask = double(props(:, label));
objMask = reshape(objMask', imsz(1), imsz(2), propNum);

