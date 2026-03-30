import os
import torch
import sys

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = self._resolve_device()
        self.Tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def _resolve_device(self):
        requested = os.environ.get("FYP_TORCH_DEVICE", "").strip().lower()
        if requested in ("", "auto"):
            if self.gpu_ids and torch.cuda.is_available():
                return torch.device('cuda:%d' % self.gpu_ids[0])
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device('cpu')
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("SimSwap requested CUDA but CUDA is not available.")
            return torch.device('cuda:%d' % (self.gpu_ids[0] if self.gpu_ids else 0))
        if requested == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not torch.backends.mps.is_available():
                raise RuntimeError("SimSwap requested MPS but MPS is not available.")
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported FYP_TORCH_DEVICE for SimSwap: {requested}")

    def _map_location(self):
        return self.device

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass
    
    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids=None):
        save_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        network.to(self.device)

    def save_optim(self, network, network_label, epoch_label, gpu_ids=None):
        save_filename = '{}_optim_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)


    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):        
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path, map_location=self._map_location()))
            except:   
                pretrained_dict = torch.load(save_path, map_location=self._map_location())                
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()                    

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    # helper loading function that can be used by subclasses
    def load_optim(self, network, network_label, epoch_label, save_dir=''):        
        save_filename = '%s_optim_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path, map_location=torch.device("cpu")))
            except:   
                pretrained_dict = torch.load(save_path, map_location=torch.device("cpu"))                
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()                    

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)                  

    def update_learning_rate():
        pass
