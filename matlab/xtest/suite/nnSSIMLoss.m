
classdef nnSSIMLoss < nntest
  methods (Test)
        function testDer(test) 
            
            % define input
            sz = 12;
%             x = single(rand([sz sz 3]));
            x = im2single(imread('E:\Flavio\datasets\immagini_esempio\Chrysanthemum.jpg'));
%             x = imresize(x,[480 640]/10);
            x = imresize(x,[15 15]);
%             x = imresize(x,[100 100]);
%             x = single(rgb2gray(x)); % test gray
            y = single(rand(size(x)));
            
            % instantiate SSIM layer
            L = 1; c1 = (0.01*L).^2;  c2 = (0.03*L).^2;
            l = dagnn.SSIMLoss('c1',c1, 'c2',c2);
            
            % call forward function to calculate the components
            l.forward({x, y});
            
            % compute derivatives with backward propagation
%             dzdy = single(ones(size(x)));
            dzdy = single(1);
            [derInputs, derParams] = l.backward({x, y}, [], dzdy);
            
            % test derivatives with incremental ratio
            delta = 1e-4;
            tau = 1e-3;
          
            % test derivatives wrt x (input)
            dzdx = derInputs{1};
            test.der(@(x) cell2mat(l.forward({x, y}, [])), x, dzdy, dzdx, delta, tau) ;
            
            % test derivatives wrt y (label)
            dzdx = derInputs{2};
%             test.der(@(y) cell2mat(l.forward({x, y}, [])), y, dzdy, dzdx, delta, tau) ;
            
        end
  end
end


