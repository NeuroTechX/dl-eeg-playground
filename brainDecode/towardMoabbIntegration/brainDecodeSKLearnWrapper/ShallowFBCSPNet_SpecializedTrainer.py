

# TODO, get this to work:


print(model)

# save/load only the model parameters(prefered solution)
torch.save(model.state_dict(), "myModel.pth")

# load the saved network (makes it possible to run bottom form same starting point 
model.load_state_dict(torch.load("myModel.pth"))

#from google.colab import files
#files.download('example.txt') 



from torch.nn import init

for param in model.conv_classifier.parameters():
    param.requires_grad = False
    
model.conv_classifier = nn.Conv2d(5, 2,(116, 1), bias=True).cuda()

optimizer = optim.Adam(model.conv_classifier.parameters(),lr=0.00006)
    
    

from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import get_balanced_batches
import torch.nn.functional as F
from numpy.random import RandomState
rng = RandomState(None)
#rng = RandomState((2017,6,30))

nb_epoch = 100
loss_rec = np.zeros((nb_epoch,2))
accuracy_rec = np.zeros((nb_epoch,2))



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10% every 30 epochs"""
    lr = 0.00006 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for i_epoch in range(nb_epoch):
    i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                            batch_size=32)
    
    
    adjust_learning_rate(optimizer,i_epoch)
    
    
    # Set model to training mode
    model.train()
    
    
    for i_trials in i_trials_in_batch:
        # Have to add empty fourth dimension to X
        batch_X = train_set.X[i_trials][:,:,:,None]
        batch_y = train_set.y[i_trials]
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        # Remove gradients of last backward pass from all parameters
        optimizer.zero_grad()
        # Compute outputs of the network
        outputs = model(net_in)
        # Compute the loss
        loss = F.nll_loss(outputs, net_target)
        # Do the backpropagation
        loss.backward()
        # Update parameters with the optimizer
        optimizer.step()

    # Print some statistics each epoch
    model.eval()
    print("Epoch {:d}".format(i_epoch))
    
    sets = {'Train' : 0, 'Test' : 1}
    for setname, dataset in (('Train', train_set), ('Test', test_set)):
        i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, batch_size=32, shuffle=False)
        outputs = []
        net_targets = []
        for i_trials in i_trials_in_batch:
            batch_X = dataset.X[i_trials][:,:,:,None]
            batch_y = dataset.y[i_trials]
            
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            net_target = var_to_np(net_target)
            output = var_to_np(model(net_in))
            outputs.append(output)
            net_targets.append(net_target)
        net_targets = np_to_var(np.concatenate(net_targets))
        outputs = np_to_var(np.concatenate(outputs))
        loss = F.nll_loss(outputs, net_targets)
 
        print("{:6s} Loss: {:.5f}".format(
            setname, float(var_to_np(loss))))
        loss_rec[i_epoch, sets[setname]] = var_to_np(loss)
    
        predicted_labels = np.argmax(var_to_np(outputs), axis=1)
        accuracy = np.mean(dataset.y  == predicted_labels)
        print("{:6s} Accuracy: {:.1f}%".format(
            setname, accuracy * 100))
        accuracy_rec[i_epoch, sets[setname]] = accuracy

