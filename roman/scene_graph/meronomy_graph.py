from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from enum import Enum
from .graph_node import GraphNode, wordnetWrapper
from .scene_graph_utils import *
from ..logger import logger
from ..map.observation import Observation
import multiprocessing
import numpy as np
import os
from ..params.data_params import ImgDataParams
from ..params.scene_graph_3D_params import SceneGraph3DParams, GraphNodeParams
from ..params.meronomy_graph_params import MeronomyGraphParams
import pickle
from ..rerun_wrapper.rerun_wrapper_window_meronomy import RerunWrapperWindowMeronomy
from robotdatapy.data.img_data import CameraParams
from robotdatapy.transform import transform
from roman.map.map import ROMANMap
from roman.object.segment import Segment
from roman.params.fastsam_params import FastSAMParams
from roman.params.system_params import SystemParams
from .scene_graph_3D_base import SceneGraph3DBase
from scipy.optimize import linear_sum_assignment
from typeguard import typechecked
from typing import Any
from .word_net_wrapper import WordWrapper, WordListWrapper

@typechecked
class MeronomyGraph(SceneGraph3DBase):

    def __init__(self, params: SystemParams, nodes: list[GraphNode], rerun_window: RerunWrapperWindowMeronomy):

        # Save Parameters
        super().__init__(params)
        self.meronomy_params: MeronomyGraphParams = params.meronomy_graph_params
        self.rerun_window: RerunWrapperWindowMeronomy = rerun_window

        # Add nodes to graph
        for node in nodes:
            self.rerun_window.update_graph(self.root_node)
            self.add_new_node_to_graph(node)
        self.rerun_window.update_graph(self.root_node)

        # Calculate the next node id
        self.next_node_ID = -1
        for node in self.root_node.get_all_descendents():
            if node.get_id() >= self.next_node_ID:
                self.next_node_ID = node.get_id() + 1

    def infer_all_relationships(self) -> None:
        graph_changed = True
        while graph_changed:
            graph_changed = False

            if self.meronomy_params.enable_synonym_merges:
                graph_changed |= self.synonym_relationship_merging()
                self.rerun_window.update_graph(self.root_node)

            if self.meronomy_params.enable_holonym_meronym_inference:
                graph_changed |= self.holonym_meronym_relationship_inference()
                self.rerun_window.update_graph(self.root_node)

            if self.meronomy_params.enable_shared_holonym_inference:
                graph_changed |= self.shared_holonym_relationship_inference()
                self.rerun_window.update_graph(self.root_node)

    class NodeRelationship(Enum):
        SHARED_HOLONYM = 0
        HOLONYM_MERONYM = 1
        SYNONYMY = 2

    def find_putative_relationships(node_list_i: list[GraphNode], node_list_j: list[GraphNode] | None = None,
                                    relationship_type: MeronomyGraph.NodeRelationship = NodeRelationship.SHARED_HOLONYM,
                                    min_cos_sim_for_synonym: float = 0.94) -> list[tuple[int, int, Any]]:
        """ 
        Finds putative relationships between two lists of nodes. 
        If second list is none, instead compute between nodes in the first list.
        """

        # Determine if comparing same list to self or two different lists
        comparing_to_self: bool = False
        if node_list_j is None:
            comparing_to_self = True
            node_list_j = node_list_i

        # Create structure to track all pairs of nodes with putative relationships
        putative_relationships: list[tuple[int, int, Any]] = []

        # Iterate through each pair of nodes
        for i, node_i in enumerate(node_list_i):
            for j, node_j in enumerate(node_list_j):

                # Skip if either node is the Root Graph Node
                if node_i.is_RootGraphNode() or node_j.is_RootGraphNode():
                    continue

                # If comparing to self, skip pairs already compared
                if comparing_to_self:
                    if i >= j: continue

                match relationship_type:
                    case MeronomyGraph.NodeRelationship.SHARED_HOLONYM:

                        # Get all holonyms for words
                        word_i_holonyms_pure: set[str] = node_i.get_all_holonyms(False)
                        word_j_holonyms_pure: set[str] = node_j.get_all_holonyms(False)

                        # TODO: Do an option where we can do upwards but excluding holonyms from a holonym/meronym relationship.

                        # Check if there is a shared holonym
                        shared_holonyms: list[str] = sorted(set.intersection(word_i_holonyms_pure, word_j_holonyms_pure))
                        if len(shared_holonyms) > 0:

                            # Make sure neither of them have a parent with the shared holonym already
                            # NOTE: This is greedy, as it may not be paired to the best parent.
                            node_i_parent = node_i.get_parent()
                            node_j_parent = node_j.get_parent()
                            if not node_i_parent.is_RootGraphNode():
                                if node_i_parent.get_words() == WordListWrapper.from_words(shared_holonyms):
                                    continue
                            if not node_j_parent.is_RootGraphNode():
                                if node_j_parent.get_words() == WordListWrapper.from_words(shared_holonyms):
                                    continue

                            # Append to list of potential nodes with shared holonyms
                            putative_relationships.append((i, j, shared_holonyms))
                            
                    case MeronomyGraph.NodeRelationship.HOLONYM_MERONYM:
                        
                        # Skip nodes that are ascendent or descendent with each other
                        if comparing_to_self and not node_i.is_descendent_or_ascendent(node_j): continue
                        
                        # Get likeliest wrapped for each node
                        word_i: WordWrapper = node_i.get_words().words[0]
                        word_j: WordWrapper = node_j.get_words().words[0]

                        logger.debug(f"Words: {word_i.word} {word_j.word}")

                        # Get all holonyms/meronyms for words
                        word_i_meronyms: set[str] = node_i.get_all_meronyms(2)
                        word_j_meronyms: set[str] = node_j.get_all_meronyms(2)
                        word_i_holonyms: set[str] = node_i.get_all_holonyms(True, 2)
                        word_j_holonyms: set[str] = word_j.get_all_holonyms(True, 2)

                        logger.debug(f"All Holonyms: {word_i_holonyms} {word_j_holonyms}")

                        # Check if there is a Holonym-Meronym relationship
                        if word_i.word in word_j_meronyms or word_i.word in word_j_holonyms or \
                           word_j.word in word_i_meronyms or word_j.word in word_i_holonyms:
                            
                            node_list_i_has_meronym: bool = False
                            if word_i.word in word_j_meronyms or word_j.word in word_i_holonyms:
                                node_list_i_has_meronym = True

                            # Append to list of potential nodes with holonym-meronym relationships
                            # with meronym coming first
                            putative_relationships.append((i, j, node_list_i_has_meronym))
                    
                    case MeronomyGraph.NodeRelationship.SYNONYMY:

                        # Get words wrapped for each node
                        words_i = node_i.get_words()
                        words_j = node_j.get_words()

                        # Calculate Semantic Similarity
                        sem_con = cosine_similarity(node_i.get_semantic_descriptor(), node_j.get_semantic_descriptor())

                        # Check for synonymy (if any of shared lemmas are same & cosine similarity is high)
                        if words_i == words_j or sem_con > min_cos_sim_for_synonym:
                            putative_relationships.append((i, j, None))

        # Return all found putative relationships
        return putative_relationships
    
    def synonym_relationship_merging(self) -> bool:
        """ Detect parent-child and nearby node synonymy merges. Return true if any merges occured. """
        any_merges_occurred = False

        # Get all putative synonymys based on WordNet & cosine similarity
        all_nodes = self.root_node.get_all_descendents()
        putative_synonyms: list[tuple[int, int, None]] = MeronomyGraph.find_putative_relationships(
                    all_nodes, relationship_type=MeronomyGraph.NodeRelationship.SYNONYMY, min_cos_sim_for_synonym=self.meronomy_params.min_cos_sim_for_synonym)
        
        # Get the list of relevant nodes in the graph
        node_list_initial = all_nodes.copy()
        
        # For each putative synonymy, calculate detected ones based on graph structure & geometry
        detected_synonyms: list[set[int]] = []
        for putative_rel in putative_synonyms:
            
            # Extract the corresponding nodes
            node_i: GraphNode = node_list_initial[putative_rel[0]]
            node_j: GraphNode = node_list_initial[putative_rel[1]]

            # If they are parent-child OR geometrically overlapping:
            _, enc_i_ratio, enc_j_ratio = self.geometric_overlap_cached(node_i, node_j)
            if node_i.is_parent_or_child(node_j) or enc_i_ratio >= self.meronomy_params.min_enc_for_synonym \
                                                 or enc_j_ratio >= self.meronomy_params.min_enc_for_synonym:
                # Add to list of detected
                detected_synonyms.append({putative_rel[0], putative_rel[1]})

        # Merge all overlaps to get final sets of all synonymys
        detected_synonyms = merge_overlapping_sets(detected_synonyms)

        # For each detected synonymy (after merging), merge all nodes one-by-one
        for detected_syn in detected_synonyms:
            any_merges_occurred = True

            # Get all nodes that are synonyms
            synonyms: list[GraphNode] = [node_list_initial[idx] for idx in detected_syn]

            # Phase 1: Merge parent-child relationships while ignoring nearby-node ones
            def merge_nodes_if_parent_and_child(node_i: GraphNode, node_j: GraphNode) -> GraphNode | None:
                """ Helper function that will merge the two nodes if they are parent & child. 
                If merge occured, returns the merged node; otherwise returns None. """

                if node_i.is_parent_or_child(node_j):
                    node_i_org_id = node_i.get_id()
                    node_j_org_id = node_j.get_id()
                    node_i_org_words = node_i.get_words()
                    node_j_org_words = node_j.get_words()

                    merged_node = node_i.merge_parent_and_child(node_j, new_id=self.next_node_ID)
                    self.next_node_ID += 1
                    self.rerun_window.update_graph(self.root_node)
                    logger.info(f"[green1]Parent-Child Synonymy[/green1]: Merging Node {node_i_org_id} ({node_i_org_words}) and Node {node_j_org_id} ({node_j_org_words}) into Node {merged_node.get_id()} ({merged_node.get_words()})")
                    return merged_node
                return None

            synonyms = merge_objs_via_function(synonyms, merge_nodes_if_parent_and_child)
    
            # Phase 2: Merge remaining nodes (non-parent/child)
            while len(synonyms) > 1:
                node_i = synonyms[0]
                node_j = synonyms[1]

                # Merge the two nodes
                merged_node = node_i.merge_with_node_meronomy(node_j, keep_children=True, new_id=self.next_node_ID)
                if merged_node is None:
                    logger.info(f"[bright_red]Merge Fail[/bright_red]: Resulting Node was invalid.")
                else:
                    # Add merged node to graph and to synonym list
                    self.next_node_ID += 1
                    self.add_new_node_to_graph(merged_node, only_leaf=True)
                    synonyms.append(merged_node)

                # Pop the previous two nodes from the synonym list
                synonyms.pop(1)
                synonyms.pop(0)

                self.rerun_window.update_graph(self.root_node)
                logger.info(f"[dark_green]Nearby Nodes Synonymy[/dark_green]: Merging Node {node_i.get_id()} ({node_i.get_words()}) and Node {node_j.get_id()} ({node_j.get_words()}) into Node {merged_node.get_id()} ({merged_node.get_words()})")
        
        return any_merges_occurred

    def holonym_meronym_relationship_inference(self) -> bool:
        """ Detect parent-child relationships between non-ascendent/descendent nodes in the graph. """
        any_inferences_occurred = False

        # Get all putative holonym-meronym relationships based on WordNet
        all_nodes = self.root_node.get_all_descendents()
        putative_holonym_meronyms: list[tuple[int, int, bool]] = MeronomyGraph.find_putative_relationships(
                    all_nodes, relationship_type=MeronomyGraph.NodeRelationship.HOLONYM_MERONYM, 
                    min_cos_sim_for_synonym=self.meronomy_params.min_cos_sim_for_synonym)

        # Get the list of nodes in the graph
        node_list_initial = all_nodes.copy()

        # For each putative relationship, calculate detected ones based on geometry
        detected_holonym_meronyms: list[tuple[int, int]] = [] # Meronym will be first
        for putative_rel in putative_holonym_meronyms:

            # Extract the corresponding nodes
            node_meronym: GraphNode = node_list_initial[putative_rel[0]]
            node_holonym: GraphNode = node_list_initial[putative_rel[1]]

            # If they are close enough
            if self.shortest_dist_to_node_size_ratio(node_meronym, node_holonym) < self.meronomy_params.ratio_dist2length_threshold_holonym_meronym:

                # Put into detected list with meronym first
                if putative_rel[2]:
                    detected_holonym_meronyms.append((putative_rel[0], putative_rel[1]))
                else:
                    detected_holonym_meronyms.append((putative_rel[1], putative_rel[0]))
       
        # Iterate over all detected holonym-meronym relationships to resolve overlaps and conflicts
        overlap = True
        while overlap:
            overlap = False

            # Map each child list index of the detected holonyms to the indices of tuples containing it
            meronym_list_index_to_holonym_list_index = defaultdict(list)
            for dhm in detected_holonym_meronyms:
                meronym_list_index_to_holonym_list_index[dhm[0]].append(dhm[1])

            # Find out if any meronyms have two or more detected holonyms
            for meronym_index, detected_holonym_indices in meronym_list_index_to_holonym_list_index.items():
                if len(detected_holonym_indices) > 1:

                    # If so, pick whichever one is closest to it geometrically
                    putative_holonyms: list[GraphNode] = [node_list_initial[idx] for idx in detected_holonym_indices]
                    meronym: GraphNode = node_list_initial[meronym_index]

                    meronym_cen: np.ndarray = meronym.get_centroid()
                    centroid_distances = [np.linalg.norm(meronym_cen - holonym.get_centroid()) for holonym in putative_holonyms]
                    holonym_to_keep_idx = centroid_distances.index(min(centroid_distances))
                    
                    # Delete all other holonym_meronym relationships
                    for idx in sorted(detected_holonym_indices, reverse=True):
                        if idx != holonym_to_keep_idx:
                            detected_holonym_meronyms.pop(idx)

                    # Our mapping has changed, so reloop
                    overlap = True
                    break

        # For each detected holonym-meronym relationship (after conflict resolution), alter the graph
        for detected_rel in detected_holonym_meronyms:
            any_inferences_occurred = True

            # Extract the corresponding nodes
            node_meronym: GraphNode = node_list_initial[detected_rel[0]]
            node_holonym: GraphNode = node_list_initial[detected_rel[1]]

            # Load words and some meronyms/holonyms
            word_holonym: WordWrapper = node_holonym.get_words().words[0]

            # Move the meronym to be child of holonym
            deleted_ids = node_meronym.remove_from_graph_complete()
            node_holonym.add_child(node_meronym)
            node_meronym.set_parent(node_holonym)

            logger.info(f"[dark_goldenrod]Holonym-Meronym Relationship Detected[/dark_goldenrod]: Node {node_meronym.get_id()} as meronym of {node_holonym.get_id()}")

            # Now that we've detected this relationship, we want to strengthen our
            # believed embeddings towards these word for holonym (since adding meronym
            # will skew holonym towards that word).
            holonym_emb: np.ndarray = wordnetWrapper.get_embedding_for_word(word_holonym.word)
            total_weight = node_holonym.get_total_weight_of_semantic_descriptors()
            node_holonym.add_semantic_descriptors([(holonym_emb, total_weight * self.meronomy_params.ratio_relationship_weight_2_total_weight)])
            if GraphNode.params.calculate_descriptor_incrementally:
                raise NotImplementedError("Holonym_Meronym Inference not currently supported with incremental semantic descriptor!")

            # Print any deleted nodes
            if len(deleted_ids) > 0:
                logger.info(f"[bright_red] Discard: [/bright_red] Node(s) {deleted_ids} removed as not enough remaining points with children removal.")
            self.rerun_window.update_graph(self.root_node)
        
        return any_inferences_occurred

    def shared_holonym_relationship_inference(self) -> bool:
        """ Detect and add higher level objects to the graph. """
        any_inferences_occurred = False

        # Get all putative holonyms based on WordNet
        all_nodes = self.root_node.get_all_descendents()
        putative_holonyms: list[tuple[int, int, list[str]]] = MeronomyGraph.find_putative_relationships(
                all_nodes, relationship_type=MeronomyGraph.NodeRelationship.SHARED_HOLONYM,
                min_cos_sim_for_synonym=self.meronomy_params.min_cos_sim_for_synonym)
        
        # Get the list of children nodes
        node_list_initial = all_nodes.copy()

        # For each putative holonym, calculate detected ones based on map geometric knowledge
        detected_holonyms: list[tuple[set[int], set[str]]] = []
        for putative_holonym in putative_holonyms:

            # Extract the corresponding nodes
            node_i: GraphNode = node_list_initial[putative_holonym[0]]
            node_j: GraphNode = node_list_initial[putative_holonym[1]]

            # If they are close enough, declare this putative holonym as detected
            if self.shortest_dist_to_node_size_ratio(node_i, node_j) < self.meronomy_params.ratio_dist2length_threshold_shared_holonym:
                detected_holonyms.append(({putative_holonym[0], putative_holonym[1]}, set(putative_holonym[2])))


        # Iterate over all detected shared holonyms to resolve overlaps and conflicts
        detected_holonyms = merge_overlapping_holonyms(detected_holonyms)
                    
        # For each detected holonym (after conflict resolution), add to graph
        for detected_holonym in detected_holonyms:

            # Extract the corresponding nodes
            nodes: list[GraphNode] = [node_list_initial[i] for i in detected_holonym[0]]

            # Calculate the first seen time  and first pose as earliest from all nodes (and children)
            earliest_node = min(nodes, key=lambda n: n.get_time_first_seen())
            first_seen = earliest_node.get_time_first_seen()
            first_pose = earliest_node.get_first_pose()

            # Create the holonym node
            holonym: GraphNode | None = GraphNode.create_node_if_possible(self.next_node_ID, 
                                            self.root_node, [], None, 0, np.zeros((0, 3), dtype=np.float64), 
                                            nodes, first_seen, 0.0, 0.0, 
                                            first_pose, np.eye(4), np.eye(4), is_RootGraphNode=False, run_dbscan=False)

            # If the node was successfully created, add to the graph
            if holonym is not None:
                self.next_node_ID += 1
                any_inferences_occurred = True

                # TODO: Maybe search to see if other nodes are in the overlap space between
                # the two children which should also be part of the set?

                for node in nodes:
                    holonym.num_sightings += node.num_sightings

                self.place_node_in_graph(holonym, self.root_node, nodes)

                # Update embedding so that it matches the word most of all!
                # TODO: If there are multiple shared holonyms, maybe update with combination of all?
                shared_holonyms: list[str] = sorted(detected_holonym[1])
                holonym_emb: np.ndarray = wordnetWrapper.get_embedding_for_word(shared_holonyms[0])
                total_weight = holonym.get_total_weight_of_semantic_descriptors()
                holonym.add_semantic_descriptors([(holonym_emb, total_weight * self.meronomy_params.ratio_relationship_weight_2_total_weight)])
                if GraphNode.params.calculate_descriptor_incrementally:
                    raise NotImplementedError("Holonym Inference not currently supported with incremental semantic descriptor!")

                # Print the detected holonym
                output_str = f"[gold1]Shared Holonym Detected[/gold1]: Node {holonym.get_id()} {shared_holonyms} from children "
                for i in range(len(nodes)):
                    output_str += f"{nodes[i].get_id()} {nodes[i].get_words()}"
                    if i + 1 < len(nodes): output_str += f", "
                logger.info(output_str)
                self.rerun_window.update_graph(self.root_node)

            else:
                raise RuntimeError("Detected Holonym is not a valid GraphNode!")
        
        return any_inferences_occurred

    def shortest_dist_to_node_size_ratio(self, a: GraphNode, b: GraphNode) -> float:
        # Get shortest distance between the nodes
        dist = self.shortest_dist_between_hulls_cached(a, b)

        # Get average of longest lines of both node
        longest_line_a = a.get_longest_line_size()
        longest_line_b = b.get_longest_line_size()
        longest_line = np.mean([longest_line_a, longest_line_b])

        # Get ratio of shortest distance to longest line (in other words, dist to object length)
        return dist / longest_line
    
    def get_nodes_as_segments(self) -> list[Segment]:

        # Convert all graph nodes to segments
        segment_list: list[Segment] = []
        relevant_nodes = self.root_node.get_all_descendents()
        for node in relevant_nodes:
            segment_list.append(node.to_segment())

        # Reset obb for each segment
        for seg in segment_list:
            seg.reset_obb()
        return segment_list