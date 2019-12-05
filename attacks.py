"""
Contains different methods for attacking models. In particular, given the gradients for token
embeddings, it computes the optimal token replacements. This code runs on CPU.
"""
from typing import Optional

import torch
import torch.jit
import numpy


# TODO: docstring average grad is steps×ids dimensional, in gpt2 there is only one step but could be
# more
@torch.no_grad()
@torch.jit.script
def hotflip_attack(
    averaged_grad,
    embedding_matrix,
    increase_loss: bool = False,
    num_candidates: int = 1,
    blacklisted_ids: Optional[torch.Tensor] = None,
):
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here
    <https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py>

    This function takes in the model's average_grad over a batch of examples and the model's token
    embedding matrix. It returns the top token candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want to
    decrease the loss of the target class (increase_loss=False).
    """
    # The result is of size seq_len×voc_size and the (i,k) coef is the (approximated) loss for using
    # the token with id k in position i while keeping the rest of the sequence constant
    gradient_dot_embedding_matrix = torch.einsum(
        "ij,kj->ik", (averaged_grad, embedding_matrix)
    )
    if not increase_loss:
        # lower versus increase the class probability.
        gradient_dot_embedding_matrix *= -1
    if blacklisted_ids is not None:
        # FIXME: should not be needed anymore
        blacklisted_ids_t = torch.jit._unwrap_optional(blacklisted_ids)
        gradient_dot_embedding_matrix[:, blacklisted_ids_t] = torch.tensor(-1e32, dtype=torch.float)
    if num_candidates > 1:  # get top k options
        best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=1)[1]
        return best_k_ids
    best_at_each_step = gradient_dot_embedding_matrix.argmax(dim=1)
    return best_at_each_step


def random_attack(embedding_matrix, trigger_token_ids, num_candidates=1):
    """
    Randomly search over the vocabulary. Gets num_candidates random samples and returns all of them.
    """
    embedding_matrix = embedding_matrix.cpu()
    new_trigger_token_ids = [
        [None] * num_candidates for _ in range(len(trigger_token_ids))
    ]
    for trigger_token_id in range(len(trigger_token_ids)):
        for candidate_number in range(num_candidates):
            # rand token in the embedding matrix
            rand_token = numpy.random.randint(embedding_matrix.shape[0])
            new_trigger_token_ids[trigger_token_id][candidate_number] = rand_token
    return new_trigger_token_ids


# steps in the direction of grad and gets the nearest neighbor vector.
def nearest_neighbor_grad(
    averaged_grad,
    embedding_matrix,
    trigger_token_ids,
    tree,
    step_size,
    increase_loss=False,
    num_candidates=1,
):
    """
    Takes a small step in the direction of the averaged_grad and finds the nearest
    vector in the embedding matrix using a kd-tree.
    """
    new_trigger_token_ids = [
        [None] * num_candidates for _ in range(len(trigger_token_ids))
    ]
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    if increase_loss:  # reverse the sign
        step_size *= -1
    for token_pos, trigger_token_id in enumerate(trigger_token_ids):
        # take a step in the direction of the gradient
        trigger_token_embed = (
            torch.nn.functional.embedding(
                torch.LongTensor([trigger_token_id]), embedding_matrix
            )
            .detach()
            .cpu()
            .numpy()[0]
        )
        stepped_trigger_token_embed = (
            trigger_token_embed
            + averaged_grad[token_pos].detach().cpu().numpy() * step_size
        )
        # look in the k-d tree for the nearest embedding
        _, neighbors = tree.query([stepped_trigger_token_embed], k=num_candidates)
        for candidate_number, neighbor in enumerate(neighbors[0]):
            new_trigger_token_ids[token_pos][candidate_number] = neighbor
    return new_trigger_token_ids
