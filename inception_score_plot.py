import matplotlib.pyplot as plt
import pickle


stuff = pickle.load(open('save.p', 'rb'))

print(stuff)

plt.plot(stuff[0])
plt.show()