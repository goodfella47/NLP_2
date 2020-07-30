import matplotlib.pyplot as plt

def plot_figures(loss,UAS_train,UAS_test,file_path):
  plt.figure(1)
  plt.plot(loss)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.title('loss')
  plt.savefig(file_path+'_loss')

  plt.figure(2)
  plt.plot(UAS_train,label="train")
  plt.plot(UAS_test,label="test")
  plt.legend(['train','test'])
  plt.title('UAS')
  plt.xlabel('epochs')
  plt.ylabel('UAS')
  plt.savefig(file_path+'UAS')


