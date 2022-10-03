# In[0]:
import time

import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import random

# populate data with the demand and distances
data = pd.read_csv('raw data/demand.csv', header=[0, 1, 2], skiprows=[3], index_col=1)
df = pd.DataFrame(data)
df = df.drop(df.columns[[0]], axis=1)
data2 = pd.read_csv('raw data/Matrix.csv', header=0, index_col="COLUMN")
df_distances = pd.DataFrame(data2)
df_distances = df_distances[df_distances.columns[~df_distances.columns.isna()]]
data_lat_lon = pd.read_csv('raw data/nyStations.csv', index_col=1)
df_lat_lon = pd.DataFrame(data_lat_lon)
df_lat_lon = df_lat_lon.drop(df_lat_lon.columns[[0]], axis=1)
merged_df = df.merge(df_lat_lon, left_index=True, right_index=True)

time_at_station = 2.5
time_per_bike = 0.5


class Station:
    # station example A :{ name : "A", demand_dict : {'15-17':-10,'18-20':5}}
    def __init__(self, name, demand_dict, latitude, longitude):
        self.name = name
        self.demand_dict = demand_dict
        self.lat = latitude
        self.lon = longitude

    def calculate_time_between_stations(self, station):
        a = int(self.name)
        b = station.name
        return df_distances.loc[a, b]

    def __repr__(self):
        return "(" + str(self.name) + "," + str(self.demand_dict) + ")"


bikeStiationList = []
time_frames_list = []
i = 0
for index, row in merged_df.iterrows():
    lat = 0
    lon = 0
    demandDict = {}
    for index, value in row.items():
        if index != 'lat' and index != 'lon':
            if i == 0:
                time_frames_list.append(index)
            demandDict[index] = value
        elif index == 'lat':
            lat = value
        else:
            lon = value
    i += 1
    bikeStiationList.append(Station(str(row.name), demandDict, lat, lon))


# In[1]:
class Fitness:

    # FitA example : { [(11 Ave & W 27 St,{'15-17': 0, '18-20': 0}),...] , '15-17'}
    def __init__(self, route, timeframe, start_station, end_station, unique_stations_demand,
                 demand_stations_visited_dictionary):
        self.fitness = 0.0
        self.start_station = start_station
        self.end_station = end_station
        self.timeframe = timeframe
        self.time_delivering = 0
        self.nr_of_stations_visited = 0
        self.nr_of_bikes_moved = 0
        self.distance_to_goal = 0
        self.count_stations_satisfied = 0
        self.demand_stations_visited_dictionary = demand_stations_visited_dictionary.copy()
        feasible_route = self.make_route_feasible(route, timeframe, start_station, end_station, unique_stations_demand)
        self.route = feasible_route

    def make_route_feasible(self, route, timeframe, start_station, end_station, usd):
        unique_stations_demand = usd
        bikes_in_van = 0
        nr_of_stations_visited = 0
        nr_of_bikes_moved = 0
        feasible_route = [start_station]
        last_station = start_station
        for station in route:
            curr_bike_station = None
            for bikeStation in unique_stations_demand:
                if bikeStation.name == station.name:
                    # this is for demand updating after action
                    curr_bike_station = bikeStation
                    break
            if curr_bike_station is not None:
                if station in self.demand_stations_visited_dictionary:
                    demand_in_timeframe = self.demand_stations_visited_dictionary[station]
                else:
                    demand_in_timeframe = curr_bike_station.demand_dict[timeframe]
                # for counting how many bike were moved
                trunk_count_before_action = bikes_in_van
                bikes_can_be_moved = True
                # move individual bike until truck full/empty or demand == 0
                while bikes_can_be_moved:
                    if demand_in_timeframe < 0 and bikes_in_van != 0:
                        demand_in_timeframe += 1
                        bikes_in_van -= 1
                    elif demand_in_timeframe > 0 and bikes_in_van != 20:
                        demand_in_timeframe -= 1
                        bikes_in_van += 1
                    else:
                        bikes_can_be_moved = False
                nr_of_stations_visited += 1
                nr_of_bikes_moved += abs(bikes_in_van - trunk_count_before_action)

                time_between_stations = last_station.calculate_time_between_stations(station) * 60
                time_spent_at_stations = nr_of_bikes_moved * time_per_bike + time_at_station * nr_of_stations_visited
                time_to_depot = 0
                if end_station is not None:
                    time_to_depot = station.calculate_time_between_stations(end_station)
                to_be_checked = self.time_delivering + time_between_stations + time_spent_at_stations + time_to_depot
                if to_be_checked < 180:
                    self.time_delivering += time_between_stations + time_spent_at_stations
                    self.demand_stations_visited_dictionary[station] = demand_in_timeframe
                    self.nr_of_stations_visited += 1
                    self.nr_of_bikes_moved += nr_of_bikes_moved
                    feasible_route.append(station)
                    last_station = station
                elif time_to_depot != 0:
                    self.time_delivering += time_to_depot
                    feasible_route.append(end_station)
                    self.evaluate_satisfaction_of_route(unique_stations_demand)
                    break
                else:
                    self.evaluate_satisfaction_of_route(unique_stations_demand)
                    break
        return feasible_route

    # returns nr_of_bikes_moved, nr_of_stations_visited, distance_to_goal and count_stations_satisfied for fitness
    # calculation in a list
    def evaluate_satisfaction_of_route(self, route):
        for bikeStation in route:
            if bikeStation in self.demand_stations_visited_dictionary:
                demand_after_distribution = self.demand_stations_visited_dictionary[bikeStation]
            else:
                demand_after_distribution = bikeStation.demand_dict[self.timeframe]
            if demand_after_distribution != 0:
                # add demand not met
                self.distance_to_goal += abs(demand_after_distribution)
            else:
                self.count_stations_satisfied += 1

    def route_fitness(self):
        results = {}
        if self.fitness == 0:
            results["nr_of_bikes_moved"] = self.nr_of_bikes_moved
            results["nr_of_stations_visited"] = self.nr_of_stations_visited
            results["distance_to_goal"] = self.distance_to_goal
            results["count_stations_satisfied"] = self.count_stations_satisfied
            results["time_delivering"] = self.time_delivering
            results["route"] = self.route
            results["demand_stations_visited_dictionary"] = self.demand_stations_visited_dictionary
            # fitness function
            self.fitness = - self.distance_to_goal
        results["fitness"] = self.fitness
        return results
    def get_route(self):
        return self.route


def create_route(station_list):
    route = []
    # 11 Ave & W 27 St used as depot & also changed it's demand to 0
    # same station no stop time
    # for i in range(random.randrange(1,4,1)):
    route += random.sample(station_list, len(station_list))
    return route


def initial_population(pop_size, station_list, timeframe):
    temp_station_list = []
    for station in station_list:
        temp_station_list.append(station)
        while abs(station.demand_dict[timeframe]) > 20:
            if station.demand_dict[timeframe] > 0:
                station.demand_dict[timeframe] -= 20
            else:
                station.demand_dict[timeframe] += 20
            temp_station_list.append(station)
    population = []
    for iterator in range(0, pop_size):
        population.append(create_route(temp_station_list))
    return population


def rank_routes(population, timeframe, start_station, end_station, unique_stations_demand,
                demand_stations_visited_dictionary):
    fitness_results = {}
    extra_fitness_results = {}
    results = []
    for i in range(0, len(population)):
        route = population[i]
        result_dict = Fitness(route, timeframe, start_station, end_station, unique_stations_demand,
                              demand_stations_visited_dictionary).route_fitness()
        fitness_results[i] = result_dict["fitness"]
        extra_fitness_results[i] = result_dict
    ranked_fitness = sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)
    results.append(ranked_fitness)
    results.append(extra_fitness_results[ranked_fitness[0][0]])
    return results


def selection(pop_ranked, elite_size):
    selection_results = []
    df_temp = pd.DataFrame(np.array(pop_ranked), columns=["Index", "Fitness"])
    df_temp['cum_sum'] = df_temp.Fitness.cumsum()
    df_temp['cum_perc'] = 100 * df_temp.cum_sum / df_temp.Fitness.sum()

    for i_elite in range(0, elite_size):
        selection_results.append(pop_ranked[i_elite][0])
    for g in range(0, len(pop_ranked) - elite_size):
        pick = 100 * random.random()
        for i_pop in range(0, len(pop_ranked)):
            if pick <= df_temp.iat[i_pop, 3]:
                selection_results.append(pop_ranked[i_pop][0])
                break
    return selection_results


def mating_pool(population, selection_results):
    matingpool = []
    for iterator in range(0, len(selection_results)):
        index_selection = selection_results[iterator]
        matingpool.append(population[index_selection])
    return matingpool


def breed(parent1, parent2):
    child_p1 = []
    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))
    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for iterator in range(start_gene, end_gene):
        child_p1.append(parent1[iterator])
    child_p2 = [item for item in parent2 if item not in child_p1]
    child = child_p1 + child_p2
    return child


def breed_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, elite_size):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            # random1out3 = random.randrange(1, 3, 1)
            # if random1out3 == 1:
            swap_with = int(random.random() * len(individual))
            try:
                station1 = individual[swapped]
                station2 = individual[swap_with]
                individual[swapped] = station2
                individual[swap_with] = station1
            except IndexError:
                print("len: " + str(len(individual)))
                print("swapped: " + str(swapped))
                print("swap with: " + str(swap_with))
    return individual


def mutate_population(population, mutation_rate):
    mutated_pop = []
    for ind in range(0, len(population)):
        mutated_ind = mutate(population[ind], mutation_rate)
        mutated_pop.append(mutated_ind)
    return mutated_pop


def next_generation(current_gen, elite_size, mutation_rate, timeframe, start_station_name, end_station_name,
                    unique_stations_demand, demand_stations_visited_dictionary):
    pop_ranked = rank_routes(current_gen, timeframe, start_station_name, end_station_name, unique_stations_demand,
                             demand_stations_visited_dictionary)[0]
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size)
    next_generation_population = mutate_population(children, mutation_rate)
    return next_generation_population


do_only_once = 0
_positive, _negative, _count_0 = 0, 0, 0


def genetic_algorithm_plot(population, pop_size, elite_size, mutation_rate, generations, unique_stations_demand,
                           nr_of_agents):
    int(round(time.time() * 1000))
    # TODO Last stop is either at the depot or at a station -> next route should start there
    timeframe = time_frames_list[2]
    start_stations_list = ["437", "396", "422", "471", "327", "2002"]
    end_stations_list = ["", "", "", "", "", ""]
    global _positive, _negative, _count_0, do_only_once
    if do_only_once == 0:
        do_only_once += 1
        positive, negative, count_0 = calculate_global_demand(population, timeframe)
        _positive, _negative, _count_0 = positive, negative, count_0
    else:
        positive, negative, count_0 = _positive, _negative, _count_0
    lowest_distance_possible = positive + negative
    highest_distance_possible = abs(positive) + abs(negative) - lowest_distance_possible

    print(count_0)
    print(negative)
    print(positive)

    se_stations_list = []
    for index_list in range(len(start_stations_list)):
        se_stations_list.append(
            return_station_with_names(start_stations_list[index_list], end_stations_list[index_list],
                                      population))
    start_station = se_stations_list[0][0]
    end_station = se_stations_list[0][1]

    pop = initial_population(pop_size, population, timeframe)
    progress = []
    nr_of_bikes_moved = []
    nr_of_stations_visited = []
    distance_to_goal = []
    count_stations_satisfied = []
    delivery_time = []

    total_progress = 0
    total_route = []
    route_objs = []
    total_nr_of_bikes_moved = 0
    total_nr_of_stations_visited = 0
    current_milli_time_agent = int(round(time.time() * 1000))
    demand_stations_visited_dict = {}

    depot_index = 1

    a_count_stations_satisfied = 0
    a_delivery_time = 0
    a_nr_of_stations_visited = 0
    a_nr_of_bikes_moved = 0
    a_progress = 0
    a_distance_to_goal = 0
    for a in range(1, nr_of_agents + 1):
        to_compare_time_agent = int(round(time.time() * 1000))
        current_milli_time_agent = to_compare_time_agent
        if a % 5 == 1 and a != 1:
            start_station = se_stations_list[depot_index][0]
            end_station = se_stations_list[depot_index][1]
            depot_index += 1
        for g in range(0, generations):
            pop = next_generation(pop, elite_size, mutation_rate, timeframe, start_station, end_station,
                                  unique_stations_demand, demand_stations_visited_dict)
            result_array = rank_routes(pop, timeframe, start_station, end_station, unique_stations_demand,
                                       demand_stations_visited_dict)
            progress_param = result_array[0]
            a_progress = progress_param[0][1]
            progress.append(a_progress)
            # print(g, end=" ")
            route_to_manipulate = result_array[1]["route"]
            route_to_print = []
            for station in route_to_manipulate:
                route_to_print.append(station.name)
            a_nr_of_bikes_moved = result_array[1]["nr_of_bikes_moved"] + total_nr_of_bikes_moved
            nr_of_bikes_moved.append(a_nr_of_bikes_moved)
            a_nr_of_stations_visited = result_array[1]["nr_of_stations_visited"] + total_nr_of_stations_visited
            nr_of_stations_visited.append(a_nr_of_stations_visited)
            a_distance_to_goal = \
                (highest_distance_possible - result_array[1]["distance_to_goal"]) / highest_distance_possible
            distance_to_goal.append(a_distance_to_goal)
            a_count_stations_satisfied = result_array[1]["count_stations_satisfied"] / len(unique_stations_demand)
            count_stations_satisfied.append(a_count_stations_satisfied)
            a_delivery_time = result_array[1]["time_delivering"]
            delivery_time.append(a_delivery_time)
            if g == generations - 1:
                total_route.append(route_to_print)
                route_objs.append(route_to_manipulate)
                demand_stations_visited_dict = result_array[1]["demand_stations_visited_dictionary"]
            to_compare_time = int(round(time.time() * 1000))
            # print(str(to_compare_time - current_milli_time), end=" ")
            current_milli_time = to_compare_time
        total_progress = a_progress
        total_nr_of_bikes_moved = a_nr_of_bikes_moved
        total_nr_of_stations_visited = a_nr_of_stations_visited
        total_delivery_time = a_delivery_time

    plot_route(route_objs, "", total_progress, "", a_count_stations_satisfied, timeframe)
    plt.plot(delivery_time)
    plt.ylabel('Delivery time / agent')
    plt.xlabel("Agent and agent's generation")
    plt.show()
    plt.plot(progress)
    plt.ylabel('Progress')
    plt.xlabel("Agent and agent's generation")
    plt.show()
    plt.plot(nr_of_bikes_moved)
    plt.ylabel('Nr of bikes moved')
    plt.xlabel("Agent and agent's generation")
    plt.show()
    plt.plot(nr_of_stations_visited)
    plt.ylabel('Nr of stations visited')
    plt.xlabel("Agent and agent's generation")
    plt.show()
    plt.plot(distance_to_goal)
    plt.ylabel('% of demand satisfied')
    plt.xlabel("Agent and agent's generation")
    plt.show()
    plt.plot(count_stations_satisfied)
    plt.ylabel('% of stations satisfied')
    plt.xlabel("Agent and agent's generation")
    plt.show()
    return a_distance_to_goal


def calculate_global_demand(population, timeframe):
    positive = 0
    negative = 0
    count_0 = 0
    for station in population:
        demand_in_timeframe = station.demand_dict[timeframe]
        if demand_in_timeframe > 0:
            positive += demand_in_timeframe
        elif demand_in_timeframe < 0:
            negative += demand_in_timeframe
        else:
            count_0 += 1
    return positive, negative, count_0


def return_station_with_names(start_station_name, end_station_name, population):
    start_station = None
    end_station = None
    for station in population:
        if station.name == start_station_name and station.name == end_station_name:
            start_station = station
            end_station = station
        elif station.name == start_station_name:
            start_station = station
        elif station.name == end_station_name:
            end_station = station
    return [start_station, end_station]


def plot_route(total_route, title, fitness, time_delivering, satisfaction_percentage, timeframe):
    start_stations_list = ["437"]
    title += " fit: " + str(fitness) + " time: " + str(time_delivering) + " satisfaction %: " + \
             str(satisfaction_percentage)
    latitude = []
    longitude = []
    for route in total_route:
        for station in route:
            latitude.append(station.lat)
            longitude.append(station.lon)
            plt.annotate(xy=[station.lon, station.lat], text=station.demand_dict[timeframe], fontsize=8)
        plt.scatter(longitude, latitude, marker='o', s=1)
        plt.plot(longitude, latitude, linewidth=0.2)
        latitude.clear()
        longitude.clear()
    for station in bikeStiationList:
        if station.name in start_stations_list:
            plt.scatter(station.lon, station.lat, c='black', s=2)
        else:
            plt.scatter(station.lon, station.lat, c='gray', s=0.2)

        # plt.scatter(lat[0], lon[0], marker='o', c='b')
    # plt.plot([lat[-1], lat[0]], [lon[-1], lon[0]], ls='--')
    plt.title(title)
    plt.axis()
    plt.savefig('filename.png', dpi=300)
    plt.show()


#
genetic_algorithm_plot(population=bikeStiationList, pop_size=100, elite_size=20, mutation_rate=0.01, generations=5,
                       unique_stations_demand=bikeStiationList, nr_of_agents=30)


def scalability_comparison(nr_of_runs_per_categ):
    runtime_data = pd.read_csv("runtime_data_results.csv", header=[0])
    df_runtime = pd.DataFrame(runtime_data)
    category_list = []
    generations_init = 10
    current_milli_time = int(round(time.time() * 1000))
    for category in range(0, 20):
        print("Category: " + str(category))
        for run in range(0, nr_of_runs_per_categ):
            print("Run: " + str(run))
            demand_satisfied = genetic_algorithm_plot(population=bikeStiationList, pop_size=generations_init,
                                                      elite_size=int(generations_init / 5),
                                                      mutation_rate=0.01, generations=50,
                                                      unique_stations_demand=bikeStiationList, nr_of_agents=5)
            to_compare_time = int(round(time.time() * 1000))
            new_row = {'algorithm': 'GA', 'population Size': generations_init, 'generations': 5,
                       'elite size': int(generations_init / 5),
                       'mutation rate': 0.01, 'runtime': to_compare_time - current_milli_time,
                       'nr of stations': len(bikeStiationList), 'nr of agents': 5, 'nr of depos': 1,
                       'timeframe': time_frames_list[2], 'demand satisfied': demand_satisfied,
                       'category': 'population_subset'}
            df_runtime = df_runtime.append(new_row, ignore_index=True)
            current_milli_time = to_compare_time
        category_list.append(generations_init)
        generations_init += 10
    df_runtime.to_csv("runtime_data_results.csv", index=False)


def plot_results_errorbars():
    runtime_data = pd.read_csv("runtime_data_results.csv", header=[0])
    df_runtime = pd.DataFrame(runtime_data)
    list_for_final_plot = []
    sum_demand_satisfied = 0
    sum_runtime = 0
    min_x_err = 999999
    max_x_err = -999999
    min_y_err = 999999
    max_y_err = -999999
    xplot = []
    yplot = []
    xerrplot = []
    yerrplot = []
    textplot = []
    changed = 0
    for index_df, row_df in df_runtime.iterrows():
        if int(index_df) % 5 == 0 and int(index_df) != 0:
            avg_demand_satisfied = sum_demand_satisfied / 5
            avg_runtime = sum_runtime / 5
            xplot.append(avg_runtime)
            yplot.append(avg_demand_satisfied)
            xerrplot.append([[avg_runtime - min_x_err], [max_x_err - avg_runtime]])
            yerrplot.append([[avg_demand_satisfied - min_y_err], [max_y_err - avg_demand_satisfied]])
            textplot.append(df_runtime.iloc[index_df - 1][2])
            min_x_err = 999999
            max_x_err = -999999
            min_y_err = 999999
            max_y_err = -999999
            sum_demand_satisfied = 0
            sum_runtime = 0
        for row_index, value_row in row_df.items():
            if row_index == "demand satisfied":
                sum_demand_satisfied += value_row
                if value_row < min_y_err:
                    min_y_err = value_row
                if value_row > max_y_err:
                    max_y_err = value_row
            if row_index == "runtime":
                sum_runtime += value_row
                if value_row < min_x_err:
                    min_x_err = value_row
                if value_row > max_x_err:
                    max_x_err = value_row

    plt.plot(xplot, yplot, label="generation_subset")
    for i in range(0, len(xplot)):
        plt.errorbar(xplot[i], yplot[i], xerr=xerrplot[i],
                     yerr=yerrplot[i], fmt="o", ecolor="red", color="red")
        plt.annotate(xy=[xplot[i], yplot[i]], text=textplot[i])
    plt.legend(loc='lower right')
    plt.ylabel('avg % demand satisfied')
    plt.xlabel("avg runtime")
    plt.savefig('test.png', dpi=300)
    plt.show()

# plot_results_errorbars()
