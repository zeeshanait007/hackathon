import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from datetime import datetime
import sys 
sys.path.append('../Shell_Code_Modified/')
import Farm_Evaluator_Vec

class WindFarmGenetic:
    elite_rate = 0.2 # elite rate: parameter for genetic algorithm
    cross_rate = 0.6 # crossover rate: parameter for genetic algorithm
    random_rate = 0.5 # random rate: parameter for genetic algorithm
    mutate_rate = 0.1 # mutation rate: parameter for genetic algorithm
    turbine = None
    pop_size = 0 # population size : how many individuals in a population
    N = 0 # number of wind turbines
    rows = 0 # how many cell rows the wind farm are divided into
    cols = 0 # how many colus the wind farm land are divided into
    iteration = 0 # how many iterations the genetic algorithm run
    NA_loc=None # not available, not usable locations index list (the index starts from 1)
    cell_width = 0  # cell width
    cell_width_half = 0  # half cell width

    # constructor of the class
    def __init__(self, rows=21, cols=21, N=0,NA_loc=None, pop_size=100, iteration=200,cell_width=0, elite_rate=0.2,
                 cross_rate=0.6, random_rate=0.5, mutate_rate=0.1):
        self.rows = rows
        self.cols = cols
        self.N = N
        self.pop_size = pop_size
        self.iteration = iteration

        self.cell_width = cell_width
        self.cell_width_half = cell_width * 0.5

        self.elite_rate = elite_rate
        self.cross_rate = cross_rate
        self.random_rate = random_rate
        self.mutate_rate = mutate_rate

        self.init_pop = None
        self.init_pop_NA = None
        self.init_pop_nonezero_indices = None
        self.NA_loc=NA_loc
        return
    
        # generate initial population
    def gen_init_pop_NA(self):
        self.init_pop,self.init_pop_NA = LayoutGridMCGenerator.gen_mc_grid_with_NA_loc(rows=self.rows, cols=self.cols,NA_loc=self.NA_loc, n=self.pop_size, N=self.N)
        self.init_pop_nonezero_indices = np.zeros((self.pop_size, self.N), dtype=np.int32)
        for ind_init_pop in range(self.pop_size):
            ind_indices = 0
            for ind in range(self.rows * self.cols):
                if self.init_pop[ind_init_pop, ind] == 1:
                    self.init_pop_nonezero_indices[ind_init_pop, ind_indices] = ind
                    ind_indices += 1
        return
    
    # load initial population
    def load_init_pop_NA(self, fname,fname_NA):
        self.init_pop = np.genfromtxt(fname, delimiter="  ", dtype=np.int32)
        self.init_pop_NA = np.genfromtxt(fname_NA, delimiter="  ", dtype=np.int32)
        self.init_pop_nonezero_indices = np.zeros((self.pop_size, self.N), dtype=np.int32)
        for ind_init_pop in range(self.pop_size):
            ind_indices = 0
            for ind in range(self.rows * self.cols):
                if self.init_pop[ind_init_pop, ind] == 1:
                    self.init_pop_nonezero_indices[ind_init_pop, ind_indices] = ind
                    ind_indices += 1
        return
    
    # save initial population
    def save_init_pop_NA(self, fname,fname_NA):
        np.savetxt(fname, self.init_pop, fmt='%d', delimiter="  ")
        np.savetxt(fname_NA, self.init_pop_NA, fmt='%d', delimiter="  ")
        return

    #MODIFIED FUNCTION for Shell Hackathon (Original: mc_gen_xy_NA)
    # generate the location index coordinate 
    # location index coordinate : in the cells, the cell with index 1 has location index (0,0) and the cell 2 has (1,0)
    # store the location index coordinate in x.dat 
    def mc_gen_xy_NA_hackathon(self, rows, cols, layouts, n, N, xfname, yfname):
        layouts_cr = np.zeros((rows * cols, 2), dtype=np.int32)  # layouts column row index
        n_copies = np.sum(layouts, axis=0)
        layouts_power = np.zeros((n, rows * cols), dtype=np.float32)
        #DEFAULT CODE
        #layouts_power = np.zeros((n, rows * cols), dtype=np.float32)
        #sum_layout_power = np.sum(layouts_power, axis=0)
        #mean_power = np.zeros(rows * cols, dtype=np.float32)
        #for i in range(rows * cols):
        #    if n_copies[i]>0:
        #        mean_power[i] = sum_layout_power[i] / n_copies[i]
        
        #specific to Shell hackathon.
        self.mc_fitness_hackathon(pop=layouts, rows=rows, cols=cols, pop_size=n, N=N, lp=layouts_power)
        sum_layout_power = np.sum(layouts_power, axis=0)
        mean_power = np.zeros(rows * cols, dtype=np.float32)
        for i in range(rows * cols):
            if n_copies[i]>0:
                mean_power[i] = sum_layout_power[i] / n_copies[i]
        #self.mc_fitness_hackathon(pop=layouts, rows=rows, cols=cols, pop_size=n, N=N,xfname=xfname) #probably not needed (Naveen)
        
        for ind in range(rows * cols):
            r_i = np.floor(ind / cols)
            c_i = np.floor(ind - r_i * cols)
            layouts_cr[ind, 0] = c_i
            layouts_cr[ind, 1] = r_i
        np.savetxt(xfname, layouts_cr, fmt='%d', delimiter="  ")
        np.savetxt(yfname, mean_power, fmt='%f', delimiter="  ")
        return
    
    #MODIFIED FUNCTION for Shell Hackathon (Original: mc_gen_xy_NA)
    # calculate fitness value of the population
    def mc_fitness_hackathon(self, pop, rows, cols, pop_size, N,lp=None,ordered_ind=None):
        fitness_val = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):
#             print("layout {}...".format(i))
            xy_position = np.zeros((N,2), dtype=np.float32)  # x y position
            cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
            ind_position = np.zeros(N, dtype=np.int32)
            ind_pos = 0
            for ind in range(rows * cols):
                if pop[i, ind] == 1:
                    r_i = np.floor(ind / cols)
                    c_i = np.floor(ind - r_i * cols)
                    cr_position[0, ind_pos] = c_i
                    cr_position[1, ind_pos] = r_i
                    xy_position[ind_pos,0] = c_i * self.cell_width + self.cell_width_half
                    xy_position[ind_pos,1] = r_i * self.cell_width + self.cell_width_half
                    ind_position[ind_pos] = ind
                    ind_pos += 1
#             print(xy_position)
            total_energy,lp_power_accum=Farm_Evaluator_Vec.get_AEP_results(turb_coords=xy_position)
            if lp is not None:
                lp[i, ind_position] = lp_power_accum
            if ordered_ind is not None:
                sorted_index = np.argsort(lp_power_accum)  # power from least to largest
                ordered_ind[i, :] = ind_position[sorted_index]
            fitness_val[i] = total_energy
        return fitness_val


    def sugga_move_worst(self, rows, cols, pop,pop_NA, pop_indices, pop_size, power_order, mars=None,svr_model=None):
        np.random.seed(seed=int(time.time()))
        for i in range(pop_size):
            r = np.random.randn()
            if r < 0.5:
                self.sugga_move_worst_case_random(i=i, rows=rows, cols=cols, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                                           pop_size=pop_size, power_order=power_order)
            else:
                self.sugga_move_worst_case_best(i=i, rows=rows, cols=cols, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                                         pop_size=pop_size, power_order=power_order, mars=mars,svr_model=svr_model)

        return

    def sugga_move_worst_case_random(self, i, rows, cols, pop,pop_NA, pop_indices, pop_size, power_order):
        np.random.seed(seed=int(time.time()))
        turbine_pos = power_order[i, 0]
        while True:
            null_turbine_pos = np.random.randint(0, cols * rows)
            if pop_NA[i, null_turbine_pos] == 0:
                break
        pop[i, turbine_pos] = 0
        pop[i, null_turbine_pos] = 1
        pop_NA[i, turbine_pos] = 0
        pop_NA[i, null_turbine_pos] = 1

        power_order[i, 0] = null_turbine_pos
        pop_indices[i, :] = np.sort(power_order[i, :])
        return

    def sugga_move_worst_case_best(self, i, rows, cols, pop,pop_NA, pop_indices, pop_size, power_order, mars,svr_model):
        np.random.seed(seed=int(time.time()))
        n_candiate = 5
        pos_candidate = np.zeros((n_candiate, 2), dtype=np.int32)
        ind_pos_candidate = np.zeros(n_candiate, dtype=np.int32)
        turbine_pos = power_order[i, 0]
        ind_can = 0
        while True:
            null_turbine_pos = np.random.randint(0, cols * rows)
            if pop_NA[i, null_turbine_pos] == 0:
                pos_candidate[ind_can, 1] = int(np.floor(null_turbine_pos / cols))
                pos_candidate[ind_can, 0] = int(np.floor(null_turbine_pos - pos_candidate[ind_can, 1] * cols))
                ind_pos_candidate[ind_can] = null_turbine_pos
                ind_can += 1
                if ind_can == n_candiate:
                    break
        svr_val = svr_model.predict(pos_candidate)
        sorted_index = np.argsort(-svr_val)  # fitness value descending from largest to least
        null_turbine_pos = ind_pos_candidate[sorted_index[0]]

        pop[i, turbine_pos] = 0
        pop[i, null_turbine_pos] = 1

        pop_NA[i, turbine_pos] = 0
        pop_NA[i, null_turbine_pos] = 1

        power_order[i, 0] = null_turbine_pos
        pop_indices[i, :] = np.sort(power_order[i, :])
        return
    
    # SUGGA select
    def sugga_select(self, pop,pop_NA, pop_indices, pop_size, elite_rate, random_rate):
        n_elite = int(pop_size * elite_rate)
        parents_ind = [i for i in range(n_elite)]
        np.random.seed(seed=int(time.time()))  # init random seed
        for i in range(n_elite, pop_size):
            if np.random.randn() < random_rate:
                parents_ind.append(i)
        parent_layouts = pop[parents_ind, :]
        parent_layouts_NA = pop_NA[parents_ind, :]
        parent_pop_indices = pop_indices[parents_ind, :]
        return len(parent_pop_indices), parent_layouts, parent_layouts_NA, parent_pop_indices
    
    # SUGGA crossover
    def sugga_crossover(self, N, pop,pop_NA, pop_indices, pop_size, n_parents,
                                     parent_layouts,parent_layouts_NA, parent_pop_indices):
        n_counter = 0
        np.random.seed(seed=int(time.time()))  # init random seed
        while n_counter < pop_size:
            male = np.random.randint(0, n_parents)
            female = np.random.randint(0, n_parents)
            if male != female:
                cross_point = np.random.randint(1, N)
                if parent_pop_indices[male, cross_point - 1] < parent_pop_indices[female, cross_point]:
                    pop[n_counter, :] = 0
                    pop[n_counter, :parent_pop_indices[male, cross_point - 1] + 1] = parent_layouts[male,
                                                                                     :parent_pop_indices[
                                                                                          male, cross_point - 1] + 1]
                    pop[n_counter, parent_pop_indices[female, cross_point]:] = parent_layouts[female,
                                                                               parent_pop_indices[female, cross_point]:]

                    pop_NA[n_counter, :] = pop[n_counter, :]
                    for i in self.NA_loc:
                        pop_NA[n_counter, i - 1] = 2

                    pop_indices[n_counter, :cross_point] = parent_pop_indices[male, :cross_point]
                    pop_indices[n_counter, cross_point:] = parent_pop_indices[female, cross_point:]
                    n_counter += 1
        return
    
    # SUGGA mutation
    def sugga_mutation(self, rows, cols, N, pop,pop_NA, pop_indices, pop_size, mutation_rate):
        np.random.seed(seed=int(time.time()))
        for i in range(pop_size):
            if np.random.randn() > mutation_rate:
                continue
            while True:
                turbine_pos = np.random.randint(0, cols * rows)
                if pop_NA[i, turbine_pos] == 1:
                    break
            while True:
                null_turbine_pos = np.random.randint(0, cols * rows)
                if pop_NA[i, null_turbine_pos] == 0:
                    break
            pop[i, turbine_pos] = 0
            pop[i, null_turbine_pos] = 1

            pop_NA[i, turbine_pos] = 0
            pop_NA[i, null_turbine_pos] = 1

            for j in range(N):
                if pop_indices[i, j] == turbine_pos:
                    pop_indices[i, j] = null_turbine_pos
                    break
            pop_indices[i, :] = np.sort(pop_indices[i, :])
        return

    # SUGGA: support vector regression guided genetic algorithm
    def sugga_genetic_alg(self, ind_time=0,svr_model=None,result_folder=None):

#         P_rate_total = self.cal_P_rate_total()
        start_time = datetime.now()
        print("Support vector regression guided genetic algorithm starts....")
        fitness_generations = np.zeros(self.iteration, dtype=np.float32)  # best fitness value in each generation
        best_layout_generations = np.zeros((self.iteration, self.rows * self.cols),
                                           dtype=np.int32)  # best layout in each generation
        best_layout_NA_generations = np.zeros((self.iteration, self.rows * self.cols),
                                              dtype=np.int32)  # best layout in each generation

        power_order = np.zeros((self.pop_size, self.N),
                               dtype=np.int32)  # each row is a layout cell indices. in each layout, order turbine power from least to largest
        pop = np.copy(self.init_pop)
        pop_NA = np.copy(self.init_pop_NA)
        pop_indices = np.copy(self.init_pop_nonezero_indices)  # each row is a layout cell indices.

        eN = int(np.floor(self.pop_size * self.elite_rate))  # elite number
        rN = int(int(np.floor(self.pop_size * self.mutate_rate)) / eN) * eN  # reproduce number
        mN = rN  # mutation number
        cN = self.pop_size - eN - mN  # crossover number

        for gen in range(self.iteration):
            print("generation {}...".format(gen))
            fitness_value = self.mc_fitness_hackathon(pop=pop, rows=self.rows, cols=self.cols, pop_size=self.pop_size,
                                                        N=self.N,
                                                        ordered_ind=power_order)
            print(fitness_value)
            sorted_index = np.argsort(-fitness_value)  # fitness value descending from largest to least

            pop = pop[sorted_index, :]
            pop_NA = pop_NA[sorted_index, :]
            power_order = power_order[sorted_index, :]
            pop_indices = pop_indices[sorted_index, :]
            if gen == 0:
                fitness_generations[gen] = fitness_value[sorted_index[0]]
                best_layout_generations[gen, :] = pop[0, :]
                best_layout_NA_generations[gen, :] = pop_NA[0, :]
            else:
                if fitness_value[sorted_index[0]] > fitness_generations[gen - 1]:
                    fitness_generations[gen] = fitness_value[sorted_index[0]]
                    best_layout_generations[gen, :] = pop[0, :]
                    best_layout_NA_generations[gen, :] = pop_NA[0, :]
                else:
                    fitness_generations[gen] = fitness_generations[gen - 1]
                    best_layout_generations[gen, :] = best_layout_generations[gen - 1, :]
                    best_layout_NA_generations[gen, :] = best_layout_NA_generations[gen - 1, :]
            self.sugga_move_worst(rows=self.rows, cols=self.cols, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                           pop_size=self.pop_size, power_order=power_order, svr_model=svr_model)



            n_parents, parent_layouts,parent_layouts_NA,  parent_pop_indices = self.sugga_select(pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                                                                       pop_size=self.pop_size,
                                                                                       elite_rate=self.elite_rate,
                                                                                       random_rate=self.random_rate)


            self.sugga_crossover(N=self.N, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices, pop_size=self.pop_size,
                                          n_parents=n_parents,
                                          parent_layouts=parent_layouts,parent_layouts_NA=parent_layouts_NA, parent_pop_indices=parent_pop_indices)



            self.sugga_mutation(rows=self.rows, cols=self.cols, N=self.N, pop=pop,pop_NA=pop_NA, pop_indices=pop_indices,
                                         pop_size=self.pop_size,
                                         mutation_rate=self.mutate_rate)

        end_time = datetime.now()
        run_time = (end_time - start_time).total_seconds()
        eta_generations = np.copy(fitness_generations)
#         eta_generations = eta_generations * (1.0 / P_rate_total)
        time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")

        filename = "{}/sugga_eta_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
        np.savetxt(filename, eta_generations, fmt='%f', delimiter="  ")
        filename = "{}/sugga_best_layouts_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
        np.savetxt(filename, best_layout_generations, fmt='%d', delimiter="  ")
        filename = "{}/sugga_best_layouts_NA_N{}_{}_{}.dat".format(result_folder,self.N, ind_time, time_stamp)
        np.savetxt(filename, best_layout_NA_generations, fmt='%d', delimiter="  ")
        print("Support vector regression guided genetic algorithm ends.")
        filename = "{}/sugga_runtime.txt".format(result_folder)
        f = open(filename, "a+")
        f.write("{}\n".format(run_time))
        f.close()
        filename = "{}/sugga_eta.txt".format(result_folder)
        f = open(filename, "a+")
        f.write("{}\n".format(eta_generations[self.iteration - 1]))
        f.close()
        return run_time, eta_generations[self.iteration - 1]

        
class LayoutGridMCGenerator:
    def __init__(self):
        return

    # rows : number of rows in wind farm
    # cols : number of columns in wind farm
    # n : number of layouts
    # N : number of turbines
    # NA_loc : not usable locations
    # generate layouts with not usable locations
    def gen_mc_grid_with_NA_loc(rows, cols, n, N,NA_loc, lofname=None,loNAfname=None):  # , xfname): generate monte carlo wind farm layout grids
        np.random.seed(seed=int(time.time()))  # init random seed
        layouts = np.zeros((n, rows * cols), dtype=np.int32)  # one row is a layout, NA loc is 0

        layouts_NA= np.zeros((n, rows * cols), dtype=np.int32)  # one row is a layout, NA loc is 2
        for i in NA_loc:
            layouts_NA[:,i-1]=2

        # layouts_cr = np.zeros((n*, 2), dtype=np.float32)  # layouts column row index
        positionX = np.random.randint(0, cols, size=(N * n * 2))
        positionY = np.random.randint(0, rows, size=(N * n * 2))
        ind_rows = 0  # index of layouts from 0 to n-1
        ind_pos = 0  # index of positionX, positionY from 0 to N*n*2-1
        # ind_crs = 0
        N_count=0
        while ind_rows < n:
            cur_state=layouts_NA[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols]
            if cur_state!=1 and cur_state!=2:
                layouts[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols]=1
                layouts_NA[ind_rows, positionX[ind_pos] + positionY[ind_pos] * cols] = 1
                N_count+=1
                if np.sum(layouts[ind_rows, :]) == N:
                    ind_rows += 1
                    N_count=0
            ind_pos += 1
            if ind_pos >= N * n * 2:
                print("Not enough positions")
                break
        # filename = "positions{}by{}by{}N{}.dat".format(rows, cols, n, N)
        if lofname is not None:
            np.savetxt(lofname, layouts, fmt='%d', delimiter="  ")
            np.savetxt(loNAfname, layouts_NA, fmt='%d', delimiter="  ")
        # np.savetxt(xfname, layouts_cr, fmt='%d', delimiter="  ")
        return layouts,layouts_NA