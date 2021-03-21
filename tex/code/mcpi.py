import numpy as np
import matplotlib.pyplot as pl
pl.style.use(['science', 'no-latex'])

def mcpi(N=1000):
    points = np.random.uniform(0,1,(2,N))
    pointsInsideIndex = np.where(np.sum(np.square(points), axis=0) <= 1)[0]
    piEstimated = pointsInsideIndex.size / N * 4.0
    return points, pointsInsideIndex, piEstimated, (piEstimated-np.pi)/np.pi
    
def main():
    N = 30000
    points, pointsInsideIndex, piEstimated, piError = mcpi(N)
    while (abs(piError*100.0) > 0.02):
        points, pointsInsideIndex, piEstimated, piError = mcpi(N)
    print(piEstimated)
    fig, ax = pl.subplots(figsize=[5,5], dpi=300)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid()    
    ax.set_title(r"Point number = %d, $\pi_e \approx$ %.4f, error = %.3f%%" %(N, piEstimated, piError*100.0), fontsize=10)
    ax.plot(points[0][pointsInsideIndex], points[1][pointsInsideIndex], 'r.', markersize=0.5)
    ax.plot(np.delete(points[0], pointsInsideIndex), np.delete(points[1], pointsInsideIndex), 'b.', markersize=0.4)
#     fig.show()
    fig.savefig("piEstimation.png", bbox_inches='tight', pad_inches=0, dpi=300)
    fig.savefig("piEstimation.pdf", bbox_inches='tight', pad_inches=0, dpi=300)
    fig.savefig("piEstimation.eps", bbox_inches='tight', pad_inches=0, dpi=300)
    pl.close()

if __name__ == "__main__":
    main()