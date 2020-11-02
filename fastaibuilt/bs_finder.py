
# from https://github.com/hal-314/fastai-batch-size-finder/blob/master/bs_finder.ipynb

from fastai.basics import *

def _lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2


def _ema_with_debias(avg, beta, yi, i):
    "Exponential moving average with debiasing"
    if avg is None: avg=0
    avg = _lin_comb(avg, yi, beta)
    return avg, avg/(1-beta**(i+1))


def _get_flatten_grads(model):
    parameters = L(model.parameters())
    grads = [param.grad.flatten().view(-1,1) for param in parameters if not type(param.grad) == type(None)]
    grad = torch.cat(grads)
    return grad




class BSFinder(Callback):
    """
    Implementation of "An Empirical Model of Large-Batch Training" article to find optimal batch size.
    It helps to find a good batch size to minimaze the training time. However, it may not be a good batch size 
    to minimize the validation loss.
    """
    run_after=Recorder
    
    def __init__(self, num_it:int=None, n_batch=5, beta=0.99, simulate_multi_gpus=True): 
        """
        num_it : the number of batches you want to process, can be set to None and it will automatically train during one epoch (or n_batch if simulate_multi_gpus is se to True)
        n_batch : the number of batches you want to store before computing the Simple Noise Scale. 20 seems to work well across different tasks.
        beta : the beta parameter for an exponential moving average to compute the sum of variances, and the scale of the gradient. If the plot is too irregular, try increasing to 0.999 or more if needed, or increase the n_batch parameter.
        simulate_multi_gpus=Simulate that user has n_batch gpus by iterating without updating model weights as original authors had. Setting it to False use DanyWind aproximation that's faster but numerically more inestable and finds a Simple Noise Scale smaller than the original Simple Noise Scale. 
        """
        store_attr(self, 'num_it, n_batch, beta, simulate_multi_gpus')

    def begin_fit(self):
        # Save original model
        self.learn.save('_tmp')
        
        if not self.num_it: self.num_it = len(self.dls.train) * (self.n_batch if self.simulate_multi_gpus else 1)
        
        self.running_scale = None
        self.running_noise = None
        
        # Use python list instead L fastai list as torch.cat doesn't understand the former
        self.stored_grads = []

        # Here, we will store the results
        self.stats = L()
        self.count=0

    def begin_validate(self): raise CancelValidException()
                
    def after_backward(self):  
        if self.train_iter >= self.num_it: raise CancelFitException()
               
        # Get gradients and store them
        self.stored_grads.append(_get_flatten_grads(self.model))
        
        self.count += 1
        if self.count != len(self.stored_grads):
            breakpoint()

        if self.simulate_multi_gpus and len(self.stored_grads) < self.n_batch: 
            self.opt.zero_grad() # Zero gradients to avoid acumulate them between batches
            #print('a', self.count, self.train_iter, learn.model.embeds[0].weight[0][:3].tolist())
            raise CancelBatchException() #skip weight update
        
        if len(self.stored_grads) == self.n_batch: 
            self.count=0
            #print('b', self.count, self.train_iter, learn.model.embeds[0].weight[0][:3].tolist())
            # We have enough batches to compute Simple Noise Scale ratio.
            
            # We concatenate the batches and empty the buffer
            stored_grads = torch.cat(self.stored_grads, dim=1)
            self.stored_grads.clear()
            
            acc_grads = stored_grads.mean(dim = 1)
        
            # The original implementation uses .mean() although in the original article didn't do it. However, averaging g_big and g_small doesn't affect to noise_scale ratio 
            if self.simulate_multi_gpus: g_small = (stored_grads ** 2).sum(dim=0).mean() 
            else: g_small = (stored_grads[:,-1] ** 2).sum() # .mean()
                
            # print((stored_grads ** 2).sum(dim=0).mean().item(), (stored_grads[:,-1] ** 2).sum().item(), (stored_grads ** 2).sum(dim=0).tolist())
            
            g_big = (acc_grads ** 2).sum() # .mean()
            
            bs = find_bs(self.yb) # or self.dls.train.bs
            b_small, b_big = bs, bs * self.n_batch
        
            noise = (b_big * g_big - b_small * g_small) / (b_big - b_small)
            scale = (g_small - g_big) / ((1 / b_small) - (1 / b_big))

            self.running_scale, scale = _ema_with_debias(self.running_scale,self.beta,scale,self.iter)
            self.running_noise, noise = _ema_with_debias(self.running_noise,self.beta,noise,self.iter)

            scale = scale.item()
            noise = noise.item()
            noise_scale = (scale / noise)
            
            # Save results
            self.stats.append(dict(n_iter=(len(self.stats)) * (1 if self.simulate_multi_gpus else self.n_batch),
                                   noise=noise, scale=scale, noise_scale=noise_scale))
        
    def after_fit(self):
        if self.train_iter < self.num_it: warn(f"Fitting was too short to complete all expectediterations. Please, increase the number of epochs")
            
        tmp_f = self.path/self.model_dir/'_tmp.pth'
        if tmp_f.exists():
            self.learn.load('_tmp')
            os.remove(tmp_f)
        
        if hasattr(self.learn, 'recorder'): 
            # index = pd.Index(torch.arange(1, len(self.stats)+1)*self.n_batch, name='n_iter')
            df = pd.DataFrame(self.stats)#, index=index)
            df.set_index('n_iter', inplace=True)
            self.recorder.bs_find_stats = df

    _docs = {"begin_fit": "Initialize container for search results and auxiliary variables and save the model",
             "after_batch": "Record hyper-parameters of this batch and potentially stop training",
             "after_backward": "Store gradients and compute Simple Noise Scale",
             "begin_validate": "Skip the validation part of training"}
@patch
def plot_bs_find(self:Recorder):
    "Plot the result of an BS Finder test (won't work if you didn't do `learn.bs_find()` before)"
    fig, ax = plt.subplots(1,1)
    stats = self.bs_find_stats
    ax.plot(stats.index, stats.noise_scale)
    ax.set_ylabel("Simple Noise Scale")
    ax.set_xlabel("# iteration")

@delegates(BSFinder)
@patch
def bs_find(self:Learner, lr, num_it=None, n_batch=5, simulate_multi_gpus=True, show_plot=True, **kwargs):
    """
    Launch a mock training to find a good batch size to minimaze training time. 
    However, it may not be a good batch size to minimize the validation loss. 
    
    A good batch size is where the Simple Noise Scale converge ignoring the small growing trend 
    with the number of iterations if exists. The optimal batch size is about an order the magnitud
    where Simple Noise scale converge. Typically,  the optimial batch size in image classification 
    problems will be 2-3 times lower where 
    """  
    num_it = num_it if num_it else len(self.dls.train)
    num_it *= n_batch if simulate_multi_gpus else 1
    n_epoch = num_it//len(self.dls.train)
    cb=BSFinder(num_it=num_it, n_batch=n_batch, simulate_multi_gpus=simulate_multi_gpus, **kwargs)
    with self.no_logging(): self.fit(n_epoch, lr, cbs=cb)
    if show_plot: self.recorder.plot_bs_find()




