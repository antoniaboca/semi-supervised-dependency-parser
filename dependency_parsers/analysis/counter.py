from dependency_parsers.data.processor import SentenceDataset
from dependency_parsers.data.params import ROOT_TOKEN, ROOT_TAG, ROOT_LABEL

class TagCounter():
    """Class that counts POS tag edges in a given dataset.
    
    Class methods:
        top_edges
        run_comparison_from_file
        get_top_edges_from_file
        tag_search_from_file
        get_statistics_from_file
    """

    def __init__(self, file=None, sentence_set=None):
        """
            The Constructor method for the TagCounter class. The Constructor accepts 
            either a file and loads the sentence set or accepts an already parsed
            sentence set.

        Args:
            file (str, optional): File to load annotated sentences from. Defaults to None.
            sentence_set (torch.utils.data.Dataset, optional): Sentence set to work with. 
                Defaults to None.

        Raises:
            Exception: User must provide at least one argument to the constructor.
            Exception: User must provide at most one argument to the constructor.
        """
        if file is None and sentence_set is None:
            raise Exception('Provide either a file to load from or a sentence set')
        
        if file is not None and sentence_set is not None:
            raise Exception('Provide exactly one argument to work with')

        self.file = file
        self.sentence_set = sentence_set
        self.loaded = True
        
        if file is not None:
            self.loaded = False

        self.tree = {}
        self.graph = {}
        
    def load(self, sentence_length=None):
        """Loads the sentence set of the conllu file specified in the Constructor.

        Args:
            sentence_length (int, optional): Maximum length of each sentence. Defaults to None.
        """

        print('Load the sentences from {}...'.format(self.file))
        self.sentence_set = SentenceDataset(self.file, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL)
        self.vocabulary = self.sentence_set.getVocabulary()
        if sentence_length is not None:
            self.sentence_set = list(filter(lambda x: len(x[0]), self.sentence_set))
        self.sentence_set = self.sentence_set.sentences
        print('Loaded the sentence set. Number of sentences: {}'.format(len(self.sentence_set)))
        self.loaded = True

    def edge_count(self, tag_type):
        """Counts the number of POS tag edges both in all target trees and in all graphs.

        Args:
            tag_type (str): Either 'xpos' or 'upos'. The type of POS tag edge to count.

        Raises:
            Exception: If a file was specified in the Constructor, the load method must 
                be called first.
            Exception: Tag type must be either 'xpos' or 'upos'.

        Returns:
            tuple: dict containing the number of occurences for each edge (both in trees
                and in graphs).
        """
        if self.loaded is False:
            raise Exception("Sentences not loaded yet.")

        if tag_type == 'xpos':
            t = 2
        elif tag_type == 'upos':
            t = 1
        else:
            raise Exception("I do not know the tag type")

        tree_edges = {}
        graph_edges = {}

        for sentence in self.sentence_set:
            for idx1 in range(len(sentence[0])):
                for idx2 in range(len(sentence[0])):
                    if idx1 == idx2:
                        continue
                    
                    xpos1 = sentence[t][idx1]
                    xpos2, parent2 = sentence[t][idx2], sentence[3][idx2]
                    
                    if (xpos1, xpos2) not in graph_edges:
                        graph_edges[(xpos1, xpos2)] = 0
                    graph_edges[(xpos1, xpos2)] += 1
                        
                    if parent2 == idx1:
                        if (xpos1, xpos2) not in tree_edges:
                            tree_edges[(xpos1, xpos2)] = 0
                        tree_edges[(xpos1, xpos2)] += 1

        self.tree, self.graph = tree_edges, graph_edges
        return tree_edges, graph_edges

    def top_edges(tree, graph, number, min_occurence):
        """Determines the top edges using the probability distribution defined in the Dissertation.

        Args:
            tree (dict): Dictionary with the total number of occurences of each edge in the target trees.
            graph (dict): Dictionary with the total number of occurences of each edge in the dependency 
                graphs.
            number (int): The number of edges to include in the top.
            min_occurence (int): The minimum number of occurences an edge must have in the dependency graphs
                to be included in the top,.

        Returns:
            tuple: A dict containing the probability of each edge in the top and a dict containing the index
                of each edge in the sorted top.
        """
        distribution = {}
        order = {}
        for edge in graph.keys():
            if graph[edge] < min_occurence:
                continue
            distribution[edge] = tree.get(edge, 0) / graph[edge]
        top = []
        for key, value in distribution.items():
            top.append((value, key))
        top.sort(reverse=True)
        distribution = {}

        idx = 0
        for value, key in top[:number]:
            distribution[key] = value
            order[key] = idx
            idx += 1
            
        return distribution, order
    
    def compare_to_prior(self, prior):
        """Given a prior, count how many edges of this dataset are covered by the prior.

        Args:
            prior (dict): A dictionary containing the edges included in the prior distribution.

        Returns:
            tuple: Counts with the number of times an edge in the dataset appears in the prior 
                distribution, both in the graphs and in the target trees.
        """
        graph_count = 0
        graph_total = 0
        tree_count = 0
        tree_total = 0
        for key, value in self.graph.items():
            if key in prior:
                graph_count += value
            graph_total += value
        
        for key, value in self.tree.items():
            if key in prior:
                tree_count += value
            tree_total += value
        
        return ((graph_count, graph_total), (tree_count, tree_total))
    
    def run_comparison(self, prior, tag_type):
        """Determine the amount of coverage this prior distribution gives.

        Args:
            prior (dict): Prior probability distribution to compare to.
            tag_type (str): Either 'xpos' or 'upos'. The type of tag to compute
                the statistics for.

        Returns:
            tuple: Ratio of edges covered, in trees and graphs, respectively.
        """
        _, _ = self.edge_count(tag_type)
        (g_count, g_total), (t_count, t_total) = self.compare_to_prior(prior)
        print('\nPercetange of tree edges in prior: {:3.3f} ({}/{})'.format(t_count/t_total, t_count, t_total))
        print('\nPercetange of graph edges in prior:{:3.3f} ({}/{})'.format(g_count/g_total, g_count, g_total))
        return t_count/t_total, g_count/g_total

    def run_comparison_from_file(file, prior, tag_type, sentence_length=None):
        """Runs the comparsion function after loading a file.

        Args:
            file (str): Conllu file to load sentences from.
            prior (dict): Prior distribution to run the comparison with.
            tag_type (str): Either 'xpos' or 'upos'. The type of tag to compute statistics for.
            sentence_length (int, optional): Whether the sentence set should contain only 
                sentences of maximum length equal to 'sentence_length'. Defaults to None.

        Returns:
            tuple: Ratio of edges covered, in trees and graphs, respectively.
        """
        counter = TagCounter(file)
        counter.load(sentence_length)
        return counter.run_comparison(prior, tag_type)
    
    def get_top_edges(self, number, min_occurence, tag_type):
        """Get the top edges from the sentence set and create a prior distribution.

        Args:
            number (int): The number of edges to include in the top.
            min_occurence (int): The minimum number of occurences an edge needs to have
                to be included in the top.
            tag_type (str): Either 'xpos' or 'upos'. The type of tag to compute statistics for.

        Returns:
            tuple: The probability for each edge in the top and its order in this top.
        """
        tree, graph = self.edge_count(tag_type)
        distribution, order = TagCounter.top_edges(tree, graph, number, min_occurence)
        
        for key, value in distribution.items():
            print('{}: {}'.format(key, value))
        return distribution, order
    
    def get_top_edges_from_file(file, number, min_occurence, tag_type, sentence_length=None):
        """Gets the top edges after loading a dataset from the file.

        Args:
            file (str): Conllu file to load sentences from.
            number (int): Number of edges to include in the top.
            min_occurence (int): Minimum number of occurences an edge must have to be
                included in the top.
            tag_type (str): Either 'xpos' or 'upos'. The type of tag to do the computation for.
            sentence_length (int, optional): Whether the sentence set should contain only 
                sentences of maximum length equal to 'sentence_length'. Defaults to None.

        Returns:
            tuple: The probability for each edge in the top and its order in this top.
        """
        counter = TagCounter(file)
        counter.load(sentence_length)
        return counter.get_top_edges(number, min_occurence, tag_type)
    
    def tag_search(self, f, min_occurence, tag_type):
        """Searches the number of edges necessary to cover a fraction of the target trees.

        Args:
            f (float): Fraction we require coverage for.
            min_occurence (int): Minimum number of occurences an edge must have to be
                included in the top.
            tag_type (str): Either 'xpos' or 'upos'. The type of tag to do the computation for.

        Returns:
            int: Number of necessary edges.
        """
        left = 1
        right = len(self.sentence_set)
        ans = 0
        while left < right:
            middle = (left + right) // 2
            dist, _ = self.get_top_edges(middle, min_occurence, tag_type)
            tree_f, _ = self.run_comparison(dist, tag_type)
            if tree_f < f:
                left = middle + 1
            else:
                ans = middle
                right = middle - 1
        
        return 
    
    def tag_iteration(self, min_occurence, tag_type):
        results = {}
        for no_edges in range(10, 310, 10):
            dist, _ = self.get_top_edges(no_edges, min_occurence, tag_type)
            f, _ = self.run_comparison(dist, tag_type)
            results[no_edges] = f
        return results
    
    def tag_iteration_from_file(file, min_occurence, tag_type, sentence_length=None):
        counter = TagCounter(file)
        counter.load(sentence_length)
        return counter.tag_iteration(min_occurence, tag_type)

    def tag_search_from_file(file, f, min_occurence, tag_type, sentence_length=None):
        """Perform a tag search after loading a sentence dataset.

        Args:
            file (str): Conllu file to load the sentences from.
            f (float): Fraction we require coverage for.
            min_occurence (int): Minimum number of occurences an edge must have to be
                included in the top.
            tag_type (str): Either 'xpos' or 'upos'. The type of tag to do the computation for.
            sentence_length (int, optional): Whether the sentence set should contain only 
                sentences of maximum length equal to 'sentence_length'. Defaults to None.

        Returns:
            int: Number of necessary edges.
        """
        counter = TagCounter(file)
        counter.load(sentence_length)
        return counter.tag_search(f, min_occurence, tag_type)  
    
    def get_statistics(self, min_occurence, tag_type, step=5):
        """Run tag search for each fraction from 0.1 to 0.95.

        Args:
            min_occurence (int): Minimum number of occurences an edge must have to be
                included in the top.
            tag_type (str): Either 'xpos' or 'upos'. The type of tag to do the computation for.
            step (int, optional): Amount to increase the fraction by. Defaults to 5.

        Returns:
            dict: Number of necessary edges for each percentage from 10% to 95%.
        """
        d = {}
        for percentage in range(10,105,5):
            d[str(percentage) + '%'] = self.tag_search(percentage/100, min_occurence, tag_type)
        return d

    def get_statistics_from_file(file, min_occurence, tag_type, sentence_length=None, step=5):
        """Runs statistics after loading sentences from a file.

        Args:
            file (str): Conllu file to load sentences from.
            min_occurence (int): Minimum number of occurences an edge must have to be
                included in the top.
            tag_type (str): Either 'xpos' or 'upos'. The type of tag to do the computation for.
            sentence_length (int, optional): Whether the sentence set should contain only 
                sentences of maximum length equal to 'sentence_length'. Defaults to None.

        Returns:
            dict: Number of necessary edges for each percentage from 10% to 95%.
        """
        counter = TagCounter(file)
        counter.load(sentence_length)
        return counter.get_statistics(min_occurence, tag_type, step)