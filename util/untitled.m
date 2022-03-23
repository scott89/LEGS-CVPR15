x=100000;
y=10;
tic
for i=1:x
rand(1,100)*rand(100,1);
end

for i=1:x
rand(1,100)*rand(100,1);
end
toc;

tic
for i=1:x
rand(1,100)*rand(100,1);
rand(1,100)*rand(100,1);
end
toc

% tic
% for i=1:x
%     for j=1:y
% rand(1,100)*rand(100,1);
% rand(1,100)*rand(100,1);
%     end
% end
% toc
% 
% tic
% for i=1:y
%     for j=1:x
% rand(1,100)*rand(100,1);
% rand(1,100)*rand(100,1);
%     end
% end
% toc

tic
for i=1:x
rand(1,100)*rand(100,1);
rand(1,100)*rand(100,1);
rand(1,100)*rand(100,1);
rand(1,100)*rand(100,1);
end
toc
tic
for i=1:x
for j=1:4
rand(1,100)*rand(100,1);
end
end
toc