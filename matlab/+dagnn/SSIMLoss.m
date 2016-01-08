classdef SSIMLoss < dagnn.Loss

    properties
        c1 = 1; c2 = 1;
        % to speed up computation in bkpr
        gaussFilt_ = {}; mx_ = {}; my_ = {}; mx2_ = {}; my2_ = {}; sx2_ = {}; sy2_ = {}; sxy_ = {}; 
        l_ = {}; cs_ = {}; ssim_ = {};
    end
  
    methods        
        function outputs = forward(obj, inputs, params)
            % reset vars
            a = cell(size(inputs{1},4),1); obj.gaussFilt_ = a; obj.mx_ = a; obj.my_ = a; obj.mx2_ = a; obj.my2_ = a; obj.sx2_ = a; obj.sy2_ = a; obj.sxy_ = a; obj.l_ = a; obj.cs_ = a; obj.ssim_ = a;
            
            % calculate ssim
            for i = 1:size(inputs{1},4)
                
                % get input and gt as x and y
                x = inputs{1}(:,:,:,i);
                y = inputs{2}(:,:,:,i);
                
                % filtraggio relu
                x = max(x, single(0));
                y = max(y, single(0));
                
                figure(10); subplot(121); imshow(x); subplot(122); imshow(y);
                
                % get gaussian filter
                gaussFilt = obj.getGaussianWeightingFilter(1.5,ndims(x));
%                 gaussFilt = fspecial('gaussian', 5, 2);
                
                % calculate mean, var and cov
                mx  = imfilter(x, gaussFilt, 'conv', 'replicate');
                my  = imfilter(y, gaussFilt, 'conv', 'replicate');
                mx2 = mx.^2;
                my2 = my.^2;
                sx2 = imfilter(x.^2, gaussFilt, 'conv', 'replicate') - mx2;
                sy2 = imfilter(y.^2, gaussFilt, 'conv', 'replicate') - my2;
                sxy = imfilter(x.*y, gaussFilt, 'conv', 'replicate') - mx.*my;
                
                % calculate l
                l = (2.*mx.*my + obj.c1) ./ (mx2 + my2 + obj.c1);
                
                % calculate cs
                cs = (2.*sxy + obj.c2) ./ (sx2 + sy2 + obj.c2);
                
                % calculate ssim
                ssim = 1 - l.*cs;
                ssim_mat = ssim;
                
                %test
%                 obj.multiplyWeightsForAmountAndSum(gaussFilt, repmat(ones(90),1,1,3), 'replicate');
                
                % store values for backpropagation
                obj.gaussFilt_{i} = gaussFilt; obj.mx_{i} = mx; obj.my_{i} = my; obj.mx2_{i} = mx2; obj.my2_{i} = my2; obj.sx2_{i} = sx2; obj.sy2_{i} = sy2; obj.sxy_{i} = sxy; obj.l_{i} = l; obj.cs_{i} = cs; obj.ssim_{i} = ssim;
                
                % calculate global ssim index
                ssim = mean(ssim(:));
                
                % fill parameters
                obj.average = obj.average + ssim;
                outputs{1}(:,:,:,i) = ssim;
            end
            obj.numAveraged = size(inputs{1},4);
            obj.average = obj.average / obj.numAveraged;
        end

        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derParams = {} ;
            for i = 1:size(inputs{1},4)
                % get input and gt as x and y
                x = inputs{1}(:,:,:,i);
                y = inputs{2}(:,:,:,i);
% 
%                 drelux_dx = x > single(0);
%                 dreluy_dy = y > single(0);
% 
%                 % filtraggio relu
%                 x = max(x, single(0));
%                 y = max(y, single(0));
%                 N = numel(x);

                % load parameters calculated in forward
                gaussFilt = obj.gaussFilt_{i}; mx = obj.mx_{i}; my = obj.my_{i}; mx2 = obj.mx2_{i}; my2 = obj.my2_{i}; sx2 = obj.sx2_{i}; sy2 = obj.sy2_{i}; sxy = obj.sxy_{i}; l = obj.l_{i}; cs = obj.cs_{i}; ssim = obj.ssim_{i};
                
                % calculate local derivatives. p is the central pixel, q a
                % generic pixel inside the patch P. dlp_dq is the
                % derivative wrt q of l evaluated in p. dl_dx is the sum of
                % these derivatives
                dl_dx = 2*obj.multiplyWeightsForAmountAndSum(gaussFilt, (my - mx.*l )./( mx2 + my2 + obj.c1), 'replicate');
                dl_dy = 2*obj.multiplyWeightsForAmountAndSum(gaussFilt, (mx - my.*l )./( mx2 + my2 + obj.c1), 'replicate');
                
                % derivative wrt q of cs evaluated in p
                dcs_dx = (2./(sx2 + sy2 + obj.c2)) .* obj.multiplyWeightsForAmountAndSum(gaussFilt, (y-my) - (x-mx).*cs, 'replicate');
                dcs_dy = (2./(sx2 + sy2 + obj.c2)) .* obj.multiplyWeightsForAmountAndSum(gaussFilt, (x-mx) - (y-my).*cs, 'replicate');
                
                % derivative of ssim
                dssim_dx = -dl_dx.*cs - l.*dcs_dx;
                dssim_dy = -dl_dy.*cs - l.*dcs_dy;
                
                %fill parameters
                derInputs{1}(:,:,:,i) = dssim_dx;
                derInputs{2}(:,:,:,i) = dssim_dy;
                
%                 derInputs{1}(:,:,:,i) = (1/numel(x)).*dssim_dx;
%                 derInputs{2}(:,:,:,i) = (1/numel(y)).*dssim_dy;
                
%                 derInputs{1}(:,:,:,i) = drelux_dx.*dssim_dx;
%                 derInputs{2}(:,:,:,i) = dreluy_dy.*dssim_dy;
            end
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1} = [1 1 1 inputSizes{1}(4)];
        end

        function obj = SSIMLoss(varargin)
            obj.load(varargin) ;
        end
        
        
    end
        % --------------------------------------------------
        
   methods (Static)
       
       function gaussFilt = getGaussianWeightingFilter(radius,N)
            % Get 2D or 3D Gaussian weighting filter

            filtRadius = ceil(radius*3); % 3 Standard deviations include >99% of the area. 
            filtSize = 2*filtRadius + 1;

            if (N < 3)
                % 2D Gaussian mask can be used for filtering even one-dimensional
                % signals using imfilter. 
                gaussFilt = fspecial('gaussian',[filtSize filtSize],radius);
            else 
                % 3D Gaussian mask
                 [x,y,z] = ndgrid(-filtRadius:filtRadius,-filtRadius:filtRadius, ...
                                -filtRadius:filtRadius);
                 arg = -(x.*x + y.*y + z.*z)/(2*radius*radius);
                 gaussFilt = exp(arg);
                 gaussFilt(gaussFilt<eps*max(gaussFilt(:))) = 0;
                 sumFilt = sum(gaussFilt(:));
                 if (sumFilt ~= 0)
                     gaussFilt  = gaussFilt/sumFilt;
                 end
            end
       end
       
       
       function out = multiplyWeightsForAmountAndSum(gaussFilt, amount, padType)
           % must implement convolution mechanism
           
           initAmt = amount;
           
           % get filter size
           gsz = size(gaussFilt);
           if numel(gsz) < 3, gsz(3) = 1; end
           
           % add padding to image
           pad = zeros([1 3]);
           for i = 1:3, pad(i) = (size(gaussFilt,i)-1) / 2; end
           oldAmount = amount;
           amount = padarray(amount, pad, padType);
           
           % create output
           out = zeros(size(amount));
           
           % iterate on each block. Pass block and filter to function
           rs = []; cs = []; chs = [];
           nmults = 0;
           for ch = 1:size(amount,3)-gsz(3)+1
                for r = 1:size(amount,1)-gsz(1)+1
                    for c = 1:size(amount,2)-gsz(2)+1
%                         rs = [rs r]; cs = [cs c]; chs = [chs ch];
                       % get center value of amount that corresponds to multiplier
                       center = [r+pad(1) c+pad(2) ch+pad(3)];
                       % get multiplier
                       m = amount(center(1), center(2), center(3));
                       % sum contributes of each pass to output
                       out(r:r+gsz(1)-1, c:c+gsz(2)-1, ch:ch+gsz(3)-1) = ...
                            out(r:r+gsz(1)-1, c:c+gsz(2)-1, ch:ch+gsz(3)-1) + m*gaussFilt;
%                         nmults = nmults + 1;
                   end
               end
           end
           
        % debug
%         dimIniziale = size(initAmt)
%         numElemIniziali = prod(size(initAmt))
%         nmults
%            [max(r(:)) max(c(:)) max(ch(:))]
            
           % remove pad
           out = out(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2), pad(3)+1:end-pad(3));
       end
       
       
    end
end


