# region Imports


import traceback
from enum import IntEnum
from itertools import chain
from multiprocessing import Process, cpu_count, Value, Queue, Array, Event
from queue import Empty
from time import sleep
from time import time
from typing import Callable, List

from numpy import mean

from base import Brain, Genome

# endregion

__all__ = ['Verbosity', 'ThreadedNeat']


# region Main run function


def _run_function(func: Callable[[Genome], float], queue: Queue, fitness_array: Array, completed_count: Value,
                  close_event: Event):
    while not close_event.is_set():
        try:
            net, index = queue.get(timeout=1)
            fitness_array[index] = func(net)
            with completed_count.get_lock():
                completed_count.value += 1
            sleep(0.01)
        except Empty:
            pass

# endregion


class Verbosity(IntEnum):
    NONE = 0
    INFO = 1
    DEBUG = 2


class ThreadedNeat(object):
    """Threaded Neat is a way to use the NeuroEvolution of Augmenting Topologies algorithm on multiple processes.

    Usage: To use you must first create the brain object and then call *.generate() to initialise it, make sure you
    specify either a max-generations or max-fitness otherwise the program will not terminate. You then need a function
    (Cannot be a lambda due to pickle limitations) which takes a Genome parameter and returns the fitness as
    a float. When these are passed to the ThreadedNeat it will repeatedly run the fitness function on the different
    Genomes until either the max-generations or max-fitness is reached.
    """

    def __init__(self, brain: Brain, func: Callable[[Genome], float], num_threads: int = cpu_count(),
                 verbosity: Verbosity = True, save_best: bool = False, save_location: str = ''):
        """Initialising a new ThreadedNeat instance

        :param brain: The brain object for creating, training and breeding Genomes. *.generate() must have been called
        :param func: The fitness function, must take a Genome as a parameter and return a float. Note: Cannot be a
        lambda due to limitations with pickle
        :param num_threads: The number of processes that will be started. Default: Number of cpu threads
        :param verbosity: How much output is displayed. Default: INFO
        :param save_best: True if you want to save the best genome from each generation. Must set save_location
        :param save_location: The location to save the best genome
        """
        self.queue: Queue = Queue(maxsize=brain.population_size)
        self.fitness: Array = Array('d', [0] * brain.population_size)
        self.brain: Brain = brain
        self.num_threads = num_threads
        self.completed: Value = Value('i', 0)
        self.close_event: Event = Event()
        self.verbosity: Verbosity = verbosity
        self.save_best: bool = save_best
        self.save_location: str = save_location
        self.threads: List[Process] = []
        self.initialised = False
        self.func: Callable[[Genome], float] = func

    def init(self):
        """Creates the processes
        init() must be called before calling start()."""
        if self.initialised:
            raise Exception("ThreadedNeat is already initialised")
        self.initialised = True

        self.threads: List[Process] = []
        for i in range(self.num_threads):
            # noinspection PyBroadException
            try:
                proc = Process(target=_run_function, args=(self.func, self.queue, self.fitness,
                                                           self.completed, self.close_event,))
                proc.daemon = True
                self.threads.append(proc)
            except Exception as e:  # If any exceptions happen when the processes are starting terminate
                self.__on_exit()
                raise e

        if self.verbosity >= Verbosity.DEBUG:
            print(f'[NEAT] {self.num_threads} threads created')

    def start(self):
        """Starts the processes and the main loop. Automatically cleans up the processes after finishing, use verbosity
        DEBUG to see the logs for the thread information
        init() must have been called first"""
        if not self.initialised:
            raise Exception("ThreadedNeat has not been initialised, call *.init()")
        start_time = time()
        # noinspection PyBroadException
        try:
            for proc in self.threads:
                proc.start()
                if self.verbosity >= Verbosity.DEBUG:
                    print(f'Process #{proc.pid} started')
            while self.brain.should_evolve():
                if self.verbosity >= Verbosity.INFO:
                    print(f'[Generation #{self.brain.generation + 1}] ', end='')
                species: List[Genome] = list(chain(self.brain.species))[0]
                number_of_elements = len(species)

                for i in range(number_of_elements):
                    self.queue.put((species[i], i))

                self.__wait_until_completed(number_of_elements)

                for i in range(number_of_elements):
                    species[i].fitness = self.fitness[i]

                if self.save_best:
                    best_genome: Genome = max(species, key=lambda s: s.fitness)
                    best_genome.save(f'{self.save_location}\\{self.brain.generation + 1}')

                if self.verbosity >= Verbosity.INFO:
                    fitness_list = [genome.fitness for genome in species]
                    print("max fitness: %.2f, mean fitness: %.2f" %
                          (max(fitness_list), float(mean(fitness_list))))

                self.brain.evolve()
        except Exception:
            print(traceback.format_exc())
        finally:
            self.__on_exit()
            if self.verbosity >= Verbosity.INFO:
                print(f"[NEAT] Finished in {round(time() - start_time, 2)}s")

    def __wait_until_completed(self, completed_count):
        """Internal function which holds the main thread until all Genomes have been evaluated"""
        while self.completed.value < completed_count:
            sleep(0.01)
        with self.completed.get_lock():
            self.completed.value = 0

    def __on_exit(self):
        """Called when the training has finished to clean up the processes"""
        self.close_event.set()
        for process in self.threads:
            if self.verbosity >= Verbosity.DEBUG:
                print(f'Joining process #{process.pid}')
            process.join()
            process.close()
        if self.verbosity >= Verbosity.DEBUG:
            print('Finished closing processes')
