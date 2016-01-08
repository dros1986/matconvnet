
classdef nnLLoss < nntest
  methods (Test)
        function testDer(test) 
            
            % define input
            sz = 1000;
            x = single(rand([sz 1]));
%             x = im2single(imread('E:\Flavio\datasets\immagini_esempio\Chrysanthemum.jpg'));
%             x = single(imresize(x,[480 640]/10));
%             x = imresize(x,[15 15]);
            y = single(rand(size(x)));
            
            % instantiate L layer
            c = 1;
            l = dagnn.LLoss('c',c);
            
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


