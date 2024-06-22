import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class cfg:

    """
    Basic configuration for BatColony class.
    """

    debug : bool = False
    benchmark : dict = {
                        "dejong":(
                                lambda x,y: x**2 + y**2,
                                (-5.12,5.12),
                            ),
                        "himmelblau":(
                            lambda x,y: (x**2 + y -11.0)**2 + (x + y**2 -7.0)**2,
                            (-5.0,5.0),
                        ),
                        "ackley":(
                            lambda x,y: -20.0*np.e**(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.e**(0.5*(np.cos(2.0*np.pi*x)+np.cos(2.0*np.pi*y))) + np.e + 20.0,
                            (-2.0,2.0),
                        ),
                        "eggcrate":(
                            lambda x,y: x**2 + y**2 + 25.0*(np.sin(x)*np.sin(x) + np.sin(y)*np.sin(y)),
                            (-5.0,5.0),
                        ),
                        }
    f_range : tuple[float,float] = (0.0,1.0)
    hp : dict = {
                "entities":10 if debug else 100,
                "timesteps":10 if debug else 100,
                "alpha":0.9,
                "gamma":0.9,
                "sleep_rate":1e-12 if debug else 1e-6,
                "verbose":False if debug else True,
                "boundaries":"closed",
                }
    plot_params : dict = {
                        "color":"#FF0000",
                        "alpha":0.7,
                        "marker":"o",
                        "s":20,
                        "edgecolors":"#000000",
                        }
    contour_cmap = matplotlib.cm.Greys_r
    contour_levels = 100
    
class BatColony:

    """
    Implements swarm-intelligence bat algorithm for a 2D setting.
    DOI: 10.4028/www.scientific.net/AMM.148-149.134
    """

    def __init__(self,
                entities : int = None,
                timesteps : int = None,
                alpha : float = None,
                gamma : float = None,
                benchmark_fn : str = None,
                sleep_rate : float = None,
                boundaries : str = None,
                random_state : int = None,
                verbose : bool = None,
                ) -> None:
    
        self.entities = entities if entities else cfg.hp["entities"]
        self.timesteps = timesteps if timesteps else cfg.hp["timesteps"]
        self.c_alpha = alpha if alpha else cfg.hp["alpha"]
        self.c_gamma = gamma if gamma else cfg.hp["gamma"]
        
        try: self.benchmark_fn = cfg.benchmark[benchmark_fn] if benchmark_fn else cfg.benchmark["dejong"]
        except: raise ValueError(f"Function '{benchmark_fn}' is not supported.")
        
        self.sleep_rate = sleep_rate if sleep_rate else cfg.hp["sleep_rate"]
        self.boundaries = boundaries if boundaries else cfg.hp["boundaries"]

        self.__validate_args()

        self._dimensions = 2
        self._freqrange = cfg.f_range
        
        self.frequency = np.zeros((self.entities,self._dimensions))
        self.velocity = np.zeros((self.entities,self._dimensions))
        self.loudness = np.ones((self.entities,))
        self.pulse_rate = np.zeros((self.entities,))

        self.__objective = self.benchmark_fn[0]

        if random_state: np.random.seed(random_state)
        self.verbose = verbose
        self.position = None

    def fill(self,) -> None:

        self.position = np.random.uniform(
                                        self.benchmark_fn[1][0],
                                        self.benchmark_fn[1][1],
                                        (self.entities,self._dimensions),
                                        )

        self.fitness = self.__get_fitness(self.position)

        fig = plt.figure()
        plt.style.use("ggplot")
        ax = plt.axes(xlim=self.benchmark_fn[1],ylim=self.benchmark_fn[1])
        X,Y = np.meshgrid(
                            np.linspace(
                                    self.benchmark_fn[1][0],
                                    self.benchmark_fn[1][1],
                                    1000),
                            np.linspace(
                                    self.benchmark_fn[1][0],
                                    self.benchmark_fn[1][1],
                                    1000),
                            )
        contour = ax.contourf(
                                X,
                                Y,
                                self.__objective(X,Y),
                                cmap=cfg.contour_cmap,
                                levels=cfg.contour_levels,
                                alpha=cfg.plot_params["alpha"]*0.75,
                                )
        self.__scatter = ax.scatter(
                                    self.position[:,0],
                                    self.position[:,1],
                                    **cfg.plot_params,
                                )

    def __validate_args(self,) -> None:

        attributes = [attr for attr in vars(self).keys() if attr not in ["boundaries","benchmark_fn"]]
        for x in attributes: assert getattr(self,x) > 0, "Values cannot be non-negative."

    def __get_fitness(self,
                    arr : np.array,
                    ) -> np.array:

        fit_fn = np.empty((self.entities,))
        for i,x in enumerate(arr):
            fit_fn[i] = self.__objective(x[0],x[1])
        return fit_fn

    def __apply_boundaries(self,
                            arr : np.array,
                            mode : str,
                            ) -> None:
        L = self.benchmark_fn[1][0]
        H = self.benchmark_fn[1][1]

        if mode == "closed":
            arr[arr < L] = L
            arr[arr > H] = H
        elif mode == "periodic":
            raise NotImplementedError() #to-do
            arr = [a if a>=L else a if a<=H else H-a if(a<L) else L+a for a in arr]
        else: raise ValueError("Acceptable values: 'closed', 'periodic'")

    def run(self,) -> None:

        if self.position is None: raise RuntimeError("Your colony is empty.")

        current_state = np.zeros((self.entities,self._dimensions))
        
        d = {
            "beta":np.random.rand(self._dimensions,),
            "epsilon":2.0*np.random.rand(self._dimensions,)-1.0,
            "alpha":self.c_alpha,
            "gamma":self.c_gamma,
            }

        self.best_fitness = np.min(self.fitness)
        best_idx = np.argmin(self.fitness)
        
        for timestep in range(self.timesteps):
            for entity in range(self.entities):

                self.frequency[entity,:] = self._freqrange[0] + (self._freqrange[1] - self._freqrange[0])*d["beta"]
                self.velocity[entity,:] = (self.position[best_idx,:] - self.position[entity,:])*self.frequency[entity,:]
                current_state[entity,:] = self.position[entity,:] + self.velocity[entity,:]

                if np.random.rand() < self.pulse_rate[entity]:
                    current_state[entity,:] += d["epsilon"]*np.mean(self.loudness)

                self.__apply_boundaries(
                                        current_state,
                                        mode=self.boundaries,
                                        )

                current_metric = self.__objective(current_state[entity,0],current_state[entity,1])

                if current_metric < self.fitness[entity]:
                        
                    self.fitness[entity] = current_metric
                    self.position[entity,:] = current_state[entity,:]
                    self.loudness[entity] *= d["alpha"]
                    self.pulse_rate[entity] *= (1.0 - np.e**(-d["gamma"]*timestep))

                    if current_metric < self.best_fitness:
                        best_idx = entity
                        self.best_fitness = current_metric

                self.__scatter.set_offsets(self.position)
                plt.draw()
                plt.pause(self.sleep_rate)
        if self.verbose:
            print(f"Best fitness achieved: {self.best_fitness:.2e}")
        plt.show()
