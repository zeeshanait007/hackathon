import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from datetime import datetime

__version__ = "1.0.0"


class SuggaFarmGenetic:
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
        self.turbine = GE_1_5_sleTurbine()
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

    # calculate total rate power
    def cal_P_rate_total(self):
        f_p = 0.0
        for ind_t in range(len(self.theta)):
            for ind_v in range(len(self.velocity)):
                f_p += self.f_theta_v[ind_t, ind_v] * self.turbine.P_i_X(self.velocity[ind_v])
        return self.N * f_p

    
    #calculate fitness value
    def sugga_fitness(self, pop, rows, cols, pop_size, N, po):
        fitness_val = np.zeros(pop_size, dtype=np.float32)
        for i in range(pop_size):

            # layout = np.reshape(pop[i, :], newshape=(rows, cols))
            xy_position = np.zeros((2, N), dtype=np.float32)  # x y position
            cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
            ind_position = np.zeros(N, dtype=np.int32)
            ind_pos = 0
            for ind in range(rows * cols):
                if pop[i, ind] == 1:
                    r_i = np.floor(ind / cols)
                    c_i = np.floor(ind - r_i * cols)
                    cr_position[0, ind_pos] = c_i
                    cr_position[1, ind_pos] = r_i
                    xy_position[0, ind_pos] = c_i * self.cell_width + self.cell_width_half
                    xy_position[1, ind_pos] = r_i * self.cell_width + self.cell_width_half
                    ind_position[ind_pos] = ind
                    ind_pos += 1
            lp_power_accum = np.zeros(N, dtype=np.float32)  # a specific layout power accumulate
            for ind_t in range(len(self.theta)):
                for ind_v in range(len(self.velocity)):
                    # print(theta[ind_t])
                    # print(np.cos(theta[ind_t]))
                    trans_matrix = np.array(
                        [[np.cos(self.theta[ind_t]), -np.sin(self.theta[ind_t])],
                         [np.sin(self.theta[ind_t]), np.cos(self.theta[ind_t])]],
                        np.float32)

                    trans_xy_position = np.matmul(trans_matrix, xy_position)
                    speed_deficiency = self.wake_calculate(trans_xy_position, N)

                    actual_velocity = (1 - speed_deficiency) * self.velocity[ind_v]
                    lp_power = self.layout_power(actual_velocity,
                                                     N)  # total power of a specific layout specific wind speed specific theta
                    lp_power = lp_power * self.f_theta_v[ind_t, ind_v]
                    lp_power_accum += lp_power

            sorted_index = np.argsort(lp_power_accum)  # power from least to largest
            po[i, :] = ind_position[sorted_index]

            fitness_val[i] = np.sum(lp_power_accum)
            #
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

        P_rate_total = self.cal_P_rate_total()
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
            fitness_value = self.sugga_fitness(pop=pop, rows=self.rows, cols=self.cols, pop_size=self.pop_size,
                                                        N=self.N,
                                                        po=power_order)
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
        eta_generations = eta_generations * (1.0 / P_rate_total)
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




