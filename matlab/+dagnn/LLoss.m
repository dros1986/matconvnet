classdef LLoss < dagnn.Loss

    properties
        c = 1;
        ti = 36;
    end
  
    methods        
        function outputs = forward(obj, inputs, params)
            
            % calculate ldist
            for i = 1:size(inputs{1},4)
                
                % get input and gt as x and y
                x = inputs{1}(:,:,:,i);
                y = inputs{2}(:,:,:,i);
                if i == obj.ti
                    figure(10); subplot(121); imshow(x); subplot(122); imshow(y);
                end
                if mod(obj.c,2)==0
                    d = (x-y).^obj.c;
                else
                    d = abs((x-y).^obj.c);
                end
                d = mean(d(:));
                
                % fill parameters
                obj.average = obj.average + d;
                outputs{1}(:,:,:,i) = d;
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
                
                %fill parameters
                if mod(obj.c,2)==0
                    derInputs{1}(:,:,:,i) =  (1/numel(x)) .* obj.c .* (x-y).^(obj.c-1);
                    derInputs{2}(:,:,:,i) = -(1/numel(y)) .* obj.c .* (x-y).^(obj.c-1);
                else
                    derInputs{1}(:,:,:,i) =  (1/numel(x)) .* obj.c .* (x-y).^(obj.c-1) .* sign((x-y).^obj.c);
                    derInputs{2}(:,:,:,i) = -(1/numel(y)) .* obj.c .* (x-y).^(obj.c-1) .* sign((x-y).^obj.c);
                end
            end
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1} = [1 1 1 inputSizes{1}(4)];
        end

        function obj = LLoss(varargin)
            obj.load(varargin) ;
        end
        
        
    end

end