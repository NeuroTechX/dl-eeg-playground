

# TODO, get this to work:

print(max(accuracy_rec[:,1]))

def smoothing(record, smoothingRadius):
 if record.shape[0] > 2 * smoothingRadius :
   record_smooth = np.zeros((record.shape[0] - (2 * smoothingRadius), record.shape[1]))
   
   for i in range(record_smooth.shape[0]):
     for j in range(record_smooth.shape[1]):
       record_smooth[i,j] = record[i:i+2*smoothingRadius,j].mean()
 
 return record_smooth


# Define smoothing radius here
smoothingRadius = 10

population_loss_rec = loss_rec
population_accuracy_rec = accuracy_rec
population_loss_smooth = smoothing(loss_rec, smoothingRadius)
population_accuracy_smooth = smoothing(accuracy_rec, smoothingRadius)


individual_loss_rec = loss_rec
individual_accuracy_rec = accuracy_rec
individual_loss_smooth = smoothing(loss_rec, smoothingRadius)
individual_accuracy_smooth = smoothing(accuracy_rec, smoothingRadius)

import matplotlib.pyplot as plt


plt.figure(figsize=(12,8))
plt.title('First training Training')
plt.subplot(221)
handles = plt.plot(population_loss_rec)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.subplot(222)
handles = plt.plot(population_loss_smooth)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.subplot(223)
handles = plt.plot(population_accuracy_rec)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.subplot(224)
handles = plt.plot(population_accuracy_smooth)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.show()

plt.figure(figsize=(12,8))
plt.title('Classifier re-training Training')
plt.subplot(221)
handles = plt.plot(individual_loss_rec)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.subplot(222)
handles = plt.plot(individual_loss_smooth)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.subplot(223)
handles = plt.plot(individual_accuracy_rec)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.subplot(224)
handles = plt.plot(individual_accuracy_smooth)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.show()






plt.figure(figsize=(12,8))
plt.subplot(221)
handles = plt.plot(population_loss_rec)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.subplot(222)
handles = plt.plot(population_loss_smooth)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.subplot(223)
handles = plt.plot(population_accuracy_rec)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.subplot(224)
handles = plt.plot(population_accuracy_smooth)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(handles, ['train', 'test'])

plt.show()
