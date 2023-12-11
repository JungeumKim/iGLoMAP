import matplotlib.pyplot as plt

def vis(Z, Y=None,axis=None, s=1, show=False,title=None,path=None,rainbow=True):

        if Y is not None:
            color = Y

        else:
            color=None

        if axis is not None:
            assert path is None, "when axis is given, we cannot save it."

        else:
            fig = plt.figure(figsize=(8, 8))
            axis = fig.add_subplot(111)



        Z0 = Z


        if color is None:
            axis.scatter(Z0[:, 0], Z0[:, 1], s=s)
        else:
            if rainbow:
                axis.scatter(Z0[:, 0], Z0[:, 1], c=color, cmap=plt.cm.gist_rainbow, s=s)
            else:
                axis.scatter(Z0[:, 0], Z0[:, 1], c=color, cmap=plt.cm.Spectral, s=s)
        axis.set_aspect('equal')
        axis.set_title(title)
        if path is not None:
            fig.savefig(path)
        if show:
            plt.show()
            plt.close()

