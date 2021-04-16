function [ s ] = slownessMap3(type)
% function for generating synthetic slowness maps
% INPUT: 'type'='ch' or 'sd' for checkerboard or smooth-discontinuous
% profle
% 12/8/18 Michael J. Bianco


if type=='ch'
    % checkerboard image
    
    ds = checkerboard(10,5,5) > 0.5; % regular checkerboard, 
    % logical array by using '>0.5' to set 1 and 0
    % 10 pixel per square, 5 tiles each in row and column, each tile has 4
    % square. Therefore each row contains 10 squares
    ds = padarray(ds,[2,2]); %shifted checkerboard
    %filled with 0s to both dimension
    ds = (ds(1:100,1:100)-0.5)/5;
    
elseif type=='sd'
    % smooth variation with discontinuity
    ds_maskBar = zeros(100,100);
    ds_maskBar(:,47:52)=1;
    %
    [xx,yy]=meshgrid(1:100,1:100);
    ds = (sin(2*pi*(xx-10)/50)+sin(2*pi*(yy+10)/50))/2;%-sqrt(2)/2; % sinusoidal slowness
    ds = ds*0.1; %scaling sinusoid
    ds_bar = ds.*ds_maskBar;
    ds = ds-ds_bar+0.1*ds_maskBar;
    %ds - ds_bar = 0
else
    disp('invalid type!')
    return
end

% adding perturbations to background speeds
s = ds+0.4;

end


