# region Imports


from __future__ import annotations

import copy
import pickle
from itertools import chain
from math import floor, ceil, tanh as python_tanh, exp
from random import uniform, choice, sample
from typing import List, Tuple, Callable

# endregion

# region Activation functions


def sigmoid(x):
    """Return the S-Curve activation of x."""
    return 1 / (1 + exp(-x))


def tanh(x):
    """Wrapper function for hyperbolic tangent activation."""
    return python_tanh(x)


def LReLU(x):
    """Leaky ReLU function for x"""
    if x >= 0:
        return x
    else:
        return 0.01 * x


# endregion

# region Functions for evaluating genes and breeding


def genomic_distance(a: Genome, b: Genome):
    """Calculate the genomic distance between two genomes."""
    a_in = set([i.innovation_id for i in a.genes])
    b_in = set([i.innovation_id for i in b.genes])

    # Does not distinguish between disjoint and excess
    matching = a_in & b_in
    disjoint = (a_in - b_in) | (b_in - a_in)
    n = len(max(a_in, b_in, key=len))

    weight_diff = 0
    for i in range(len(matching)):
        if a.genes[i].innovation_id in matching:
            weight_diff += abs(a.genes[i].weight - b.genes[i].weight)

    return len(disjoint) / n + weight_diff / len(matching)


def genomic_crossover(a: Genome, b: Genome):
    """Breed two genomes and return the child. Matching genes
    are inherited randomly, while disjoint genes are inherited
    from the fitter parent.
    """
    a_in = set([i.innovation_id for i in a.genes])
    b_in = set([i.innovation_id for i in b.genes])

    # Template genome for child
    child: Genome = Genome(a.input_count, a.output_count, a.innovations)

    # Inherit homologous gene from a random parent
    for i in a_in & b_in:
        parent: Genome = choice([a, b])
        for g in parent.genes:
            if g.innovation_id == i:
                child.genes.append(g)

    # Inherit disjoint/excess genes from fitter parent
    disjoint_a = [i for i in a.genes if i.innovation_id in a_in - b_in]
    disjoint_b = [i for i in b.genes if i.innovation_id in b_in - a_in]

    if a.fitness > b.fitness:
        child.genes.extend(disjoint_a)
    elif b.fitness > a.fitness:
        child.genes.extend(disjoint_b)
    else:
        for d in (disjoint_a + disjoint_b):
            if uniform(0, 1) > 0.5:
                child.genes.append(d)

    child.nodes = max(chain.from_iterable(child.edges))
    child.reset()
    return child


def breed(species: List[Genome]) -> Genome:
    """Return a child as a result of either a mutated clone or crossover between two parent genomes

    :param species: The list of genomes which will be used to create the child
    :return The child genome"""
    if uniform(0, 1) < 0.4 or len(species) == 1:
        child = choice(species).clone()
        child.mutate()
    else:
        parent_a, parent_b = sample(species, k=2)
        child = genomic_crossover(parent_a, parent_b)

    return child


# endregion


class Gene(object):
    def __init__(self, innovation_id: int, weight: float, enabled: bool):
        self.__innovation_id: int = innovation_id
        self.weight: float = weight
        self.enabled: bool = enabled

    # Getter for innovation id, it shouldn't be set
    @property
    def innovation_id(self) -> int:
        return self.__innovation_id


class Genome(object):
    """Base class for a standard genome used by the NEAT algorithm"""

    def __init__(self, input_count: int, output_count: int, innovations: List[Tuple[int, int]]):
        """Construct a new 'Genome' object

        :param input_count: The number of input neurons
        :param output_count: The number of output neurons
        :param innovations: A list of pairs of neurons"""

        self.__input_count: int = input_count
        self.__output_count: int = output_count

        self.__unhidden: int = input_count + output_count
        self.__max_node: int = self.__unhidden
        self.__activations: List[float] = []

        # Structure
        self.__genes: List[Gene] = []
        self.__innovations: List[Tuple[int, int]] = innovations

        # Performance
        self.__fitness: float = 0
        self.__adjusted_fitness: float = 0

    def generate(self):
        """Generate the neural network of this genome with minimal
        initial topology, i.e. (no hidden nodes). Call on genome
        creation.
        """
        for i in range(self.__input_count):
            for j in range(self.__input_count, self.__unhidden):
                self.add_edge(i + 1, j + 1)

        self.reset()

    def forward(self, inputs: List[float], activation: Callable[[float], float] = sigmoid) -> List[float]:
        """Evaluate inputs and calculate the outputs of the
        neural network via the forward propagation algorithm.
        """
        if len(inputs) < self.__input_count:
            raise ValueError(f"Expected {self.__input_count} inputs, got {len(inputs)}")

        # Set input activations
        i: int
        for i in range(self.__input_count):
            self.__activations[i] = inputs[i]

        # Iterate through edges and perform forward propagation algorithm
        # Node Sort: INPUT -> HIDDEN -> OUTPUT (exclude INPUT though)
        nodes = chain(range(self.__unhidden, self.__max_node),
                      range(self.__input_count, self.__unhidden))

        n: int
        for (n) in nodes:
            total = 0
            # OUTPUT = activation(sum(weight*INPUT))
            for gene in self.__genes:
                (__in, __out) = self.__innovations[gene.innovation_id]
                weight = gene.weight
                enabled = gene.enabled
                if n + 1 == __out and enabled:
                    # Add all incoming connections to this node * their weights
                    total += weight * self.__activations[__in - 1]

            self.__activations[n] = activation(total)

        return [self.__activations[n - 1] for n in range(1, self.__max_node + 1) if self.is_output(n)]

    def mutate(self):
        """Randomly mutate the genome to initiate variation"""
        if self.is_disabled:
            self.add_enabled()

        rand = uniform(0, 1)
        if rand < 0.3:
            self.add_node()
        elif 0.3 <= rand < 0.8:
            pair: Tuple[int, int] = self.random_pair()
            if pair not in self.edges:
                self.add_edge(*pair)
            else:
                self.shift_weight()
        else:
            self.shift_weight()

        self.reset()

    def add_node(self):
        """Add a new node between a randomly selected edge,
        disabling the parent edge.
        """
        self.__max_node += 1
        enabled: List[Gene] = [g for g in self.__genes if g.enabled]
        gene = choice(enabled)
        gene.enabled = False
        i, j = self.__innovations[gene.innovation_id]

        self.add_edge(i, self.__max_node)
        self.add_edge(self.__max_node, j)

    def add_edge(self, i, j):
        """Add a new connection between existing nodes."""
        # Ignore if already present
        if (i, j) in self.edges:
            return

        # Add it to the edge database
        if (i, j) not in self.__innovations:
            self.__innovations.append((i, j))

        # Update this genome's genes
        inv = self.__innovations.index((i, j))
        self.__genes.append(Gene(inv, uniform(-1, 1), True))

    def add_enabled(self):
        """Re-enable a random disabled gene."""
        disabled: List[Gene] = [g for g in self.__genes if not g.enabled]

        if len(disabled) > 0:
            choice(disabled).enabled = True

    def shift_weight(self):
        """Randomly shift, perturb, or set one of the edge weights."""
        # Shift a randomly selected weight
        gene: Gene = choice(self.__genes)
        rand = uniform(0, 1)
        if rand <= 0.2:
            # Perturb
            gene.weight += 0.1 * choice([-1, 1])
        elif 0.2 < rand <= 0.5:
            # New random value
            gene.weight = uniform(-1, 1)
        else:
            # Reflect
            gene.weight *= -1

        # Keep within [-1.0, 1.0]
        if gene.weight < 0:
            gene.weight = max(-1.0, gene.weight)
        else:
            gene.weight = min(1.0, gene.weight)

    def random_pair(self) -> Tuple[int, int]:
        """Generate random nodes (i, j) such that:
        1. i < j
        2. i is not an output
        3. j is not an input

        Ensures a directed acyclic graph (DAG).

        :return A tuple containing a pair of nodes (from, to)
        """
        i: int = choice([n for n in range(1, self.__max_node + 1) if not self.is_output(n)])
        j_list: List[int] = [n for n in range(i + 1, self.__max_node + 1) if not self.is_input(n) and n > i]

        if len(j_list) == 0:
            self.add_node()
            j = self.__max_node
        else:
            j = choice(j_list)

        return i, j

    def is_input(self, node: int) -> bool:
        """Determine if a node is an input

        :param node: The node index
        :return True if the node is an input node otherwise False"""

        return node <= self.__input_count

    def is_output(self, node: int) -> bool:
        """Determine if a node is an output

        :param node: The node index
        :return True if the node is an output node otherwise False"""

        return self.__input_count < node <= self.__unhidden

    @property
    def is_disabled(self) -> bool:
        """Determine if all of its genes are disabled

        :return True if all genes are disabled otherwise False"""

        return all(not g.enabled for g in self.__genes)

    @property
    def fitness(self) -> float:
        """Get the fitness of the genome

        :return The fitness"""
        return self.__fitness

    @fitness.setter
    def fitness(self, fitness):
        """Set the fitness score of this genome

        :param fitness: The fitness score"""
        self.__fitness = fitness

    @property
    def genes(self) -> List[Gene]:
        """Return this genome's genes

        :return List of genes"""
        return self.__genes

    @property
    def innovations(self) -> List[Tuple[int, int]]:
        """Get this genome's innovation database

        :return The list of innovations"""
        return self.__innovations

    @property
    def nodes(self) -> int:
        """Get the number of nodes in the network

        :return The amount of nodes"""
        return self.__max_node

    @nodes.setter
    def nodes(self, nodes):
        """Set the number of nodes in the network

        :param nodes: The number of nodes"""
        self.__max_node = nodes

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Generate the network's edges given it's innovation numbers

        :return List of edges"""
        return [self.__innovations[g.innovation_id] for g in self.__genes]

    @property
    def adjusted_fitness(self):
        """Get the adjusted fitness of the genome

        :return The adjusted fitness"""
        return self.__adjusted_fitness

    @adjusted_fitness.setter
    def adjusted_fitness(self, fitness):
        """Set the adjusted fitness of the genome

        :param fitness: The adjusted fitness"""
        self.__adjusted_fitness = fitness

    @property
    def input_count(self):
        """Get the number of input neurons

        :return The number of input neurons"""
        return self.__input_count

    @property
    def output_count(self):
        """Get the number of output neurons

        :return The number of output neurons"""
        return self.__output_count

    def reset(self):
        """Reset the genome's activation and fitness values."""
        self.__activations = [0 for _ in range(self.__max_node)]
        self.__fitness = 0

    def clone(self):
        """Return a clone of the genome, maintaining internal
        reference to global innovation database.

        :return A copy of the Genome
        """
        # DON'T FORGET TO UPDATE INTERNAL REFERENCES TO OTHER OBJECTS WHEN CLONING
        clone: Genome = copy.deepcopy(self)
        clone.__innovations = self.__innovations
        return clone

    def save(self, filename: str):
        """Save an instance of this genome to disk."""
        with open(filename + '.genome', 'wb') as _out:
            pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename: str) -> Genome:
        """Load an instance of a genome from disk"""
        with open(filename + '.genome', 'rb') as _in:
            return pickle.load(_in)


class Brain(object):
    """Base class for a 'brain' that learns through evolution of a population of genomes"""

    def __init__(self, input_count: int, output_count: int, population: int = 100, max_fitness: float = -1,
                 max_generations: int = -1, delta_threshold: float = 3, cull_percent: float = 0.75):
        """Construct a new 'Genome' object

        :param input_count: The number of input neurons
        :param output_count: The number of output neurons
        :param population: The number of genomes per generation
        :param max_fitness: The maximum fitness, if it is achieved the system stops evolving
        :param max_generations: The maximum number of generations, if this is achieved the system stops evolving
        :param delta_threshold: The minimum difference in two Genomes needed to classify as two species
        :param cull_percent: What percentage of the current species are culled"""

        self.__input_count: int = input_count
        self.__output_count: int = output_count
        self.__edges: List[Tuple[int, int]] = []

        self.__species: List[List[Genome]] = []
        self.__population: int = population

        self.__delta_threshold: float = delta_threshold
        self.__cull_percent: float = cull_percent

        self.__generation: int = 0
        self.__max_generations: int = max_generations

        self.__current_species: int = 0
        self.__current_genome: int = 0

        self.__max_fitness: float = max_fitness
        self.__global_max_fitness: float = 0
        self.__fitness_sums: List[float] = []

    def generate(self):
        """Generate the initial population of genomes"""
        for i in range(self.__population):
            g: Genome = Genome(self.__input_count, self.__output_count, self.__edges)
            g.generate()
            self.classify_genome(g)

    def classify_genome(self, genome: Genome):
        """Classify genomes into species via the genomic distance algorithm

        :param genome: The genome to classify"""
        if len(self.__species) == 0:
            # There are no species so create a new one
            self.__species.append([genome])
        else:
            # Compare genome against representative s[0] in each species
            for species in self.__species:
                rep = species[0]
                if genomic_distance(genome, rep) < self.__delta_threshold:
                    species.append(genome)
                    return
            self.__species.append([genome])

    def update_fitness(self):
        """Update the adjusted fitness values of each genome"""
        self.__fitness_sums = []
        species: List[Genome]
        for species in self.__species:
            if self.population == len(species):
                sh: int = 1
            else:
                sh: int = self.population - len(species)

            genome: Genome
            for genome in species:
                genome.adjusted_fitness = genome.fitness / float(sh)

            self.__fitness_sums.append(sum([genome.adjusted_fitness for genome in species]))

    def update_fittest(self):
        """Update the highest fitness score of the whole population"""
        best: List[float] = []
        for species in self.__species:
            best.append(max(map(lambda g: g.fitness, species)))

        if max(best) > self.__global_max_fitness:
            self.__global_max_fitness = max(best)

    def evolve(self):
        """Evolve the population by eliminating the poorest performing
        genomes and repopulating with mutated children, prioritizing
        the most promising species.
        """
        self.update_fitness()
        global_fitness_sum = float(sum(self.__fitness_sums))

        if global_fitness_sum == 0:
            # No progress, mutate everybody
            for species in self.__species:
                for genome in species:
                    genome.mutate()

        else:
            # 1. Eliminate lowest performing genomes per species
            # 2. Repopulate
            self.cull_genomes(False)

            children = []
            for idx, species in enumerate(self.__species):
                ratio = self.__fitness_sums[idx] / global_fitness_sum
                offspring = floor(ratio * (self.__population - self.population))

                for j in range(int(offspring)):
                    children.append(breed(species))

            for genome in children:
                self.classify_genome(genome)

        self.__generation += 1

    def cull_genomes(self, fittest_only: bool):
        """Exterminate the weakest genomes per species

        :param fittest_only: If true keep only the fittest genome, otherwise remove based on cull percent"""
        for idx, species in enumerate(self.__species):
            if fittest_only:
                remaining = len(species) - 1
            else:
                remaining = int(ceil(self.__cull_percent * len(species)))

            culled: List[Genome] = sorted(species, key=lambda genome: genome.fitness)[remaining - 1:]
            new_rep: Genome = min(culled, key=lambda g: genomic_distance(g, species[0]))

            self.__species[idx] = [new_rep] + [genome for genome in culled if genome != new_rep]

    def should_evolve(self):
        """Determine if the system should continue to evolve
        based on the maximum fitness and generation count.
        """
        self.update_fittest()
        fit = self.__global_max_fitness != self.__max_fitness
        end = self.__generation != self.__max_generations

        return fit and end

    def next_iteration(self):
        """Call after every evaluation of individual genomes to
        progress training.
        """
        if self.__current_genome < len(self.__species[self.__current_species]) - 1:
            self.__current_genome += 1
        else:
            if self.__current_species < len(self.__species) - 1:
                self.__current_species += 1
                self.__current_genome = 0
            else:
                # Evolve to the next generation
                self.evolve()
                self.__current_species = 0
                self.__current_genome = 0

    @property
    def fittest(self):
        """Get the highest fitness score of the whole population."""
        return self.__global_max_fitness

    @property
    def population_size(self):
        return self.__population

    @property
    def population(self):
        """Return the true (calculated) population size."""
        return sum([len(s) for s in self.__species])

    @property
    def current(self):
        """Get the current genome for evaluation."""
        return self.__species[self.__current_species][self.__current_genome]

    @property
    def current_species(self):
        """Get index of current species being evaluated."""
        return self.__current_species

    @property
    def current_genome(self):
        """Get index of current genome being evaluated."""
        return self.__current_genome

    @property
    def generation(self):
        """Get the current generation number of this population."""
        return self.__generation

    @property
    def species(self):
        """Get the list of species and their respective member genomes."""
        return self.__species

    @property
    def innovations(self):
        """Get this population's innovation database."""
        return self.__edges

    def save(self, filename):
        """Save an instance of the population to disk."""
        with open(filename + '.neat', 'wb') as _out:
            pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """Return an instance of a population from disk."""
        with open(filename + '.neat', 'rb') as _in:
            return pickle.load(_in)
