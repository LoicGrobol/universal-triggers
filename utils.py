from collections import namedtuple
import dataclasses
import typing
from typing import List

from operator import itemgetter
from copy import deepcopy
import heapq
import numpy
import torch
import torch.jit
import torch.optim as optim
from allennlp.common.util import lazy_groups_of
from allennlp.data.iterators import BucketIterator
from allennlp.nn.util import move_to_device
from allennlp.modules.text_field_embedders import TextFieldEmbedder


T = typing.TypeVar("T")


@dataclasses.dataclass(order=True)
class BeamHeapElem:
    heap_key: typing.Any
    set_key: typing.Hashable
    value: typing.Any = dataclasses.field(compare=True)


class Beam(typing.MutableSet[T]):
    """Works as both a set and a heap and optionally limit size.

    `set_key` is assumed to be finer than `heap_key`, i.e.
    `set_key(x) == set_key(y)` ⇒ `heap_key(x) == heap_key(y)`
    """

    def __init__(
        self,
        itr: typing.Optional[typing.Iterable[T]] = None,
        heap_key: typing.Optional[typing.Callable[[T], typing.Any]] = None,
        set_key: typing.Optional[typing.Callable[[T], typing.Hashable]] = None,
        size_limit: typing.Optional[int] = None,
    ):
        self._set: typing.Set[typing.Hashable]
        self._heap: typing.List[BeamHeapElem]
        self.heap_key: typing.Callable[[T], typing.Any]
        self.set_key: typing.Callable[[T], typing.Hashable]
        self.size_limit: typing.Optional[int]
        if isinstance(itr, Beam):
            self.heap_key = heap_key if heap_key is not None else itr.heap_key
            self.set_key = set_key if set_key is not None else itr.set_key
            self.size_limit = size_limit if size_limit is not None else itr.size_limit
            self._set = set(itr._set)
            self._heap = list(itr._heap)
        else:
            self.heap_key = heap_key if heap_key is not None else lambda x: x
            if set_key is not None:
                self.set_key = set_key
            else:
                self.set_key = typing.Cast(typing.Callable[[T], typing.Hashable], hash)
            self.size_limit = size_limit
            self._set = set()
            self._heap = []
            if itr is not None:
                self.update(itr)

    def peek(self):
        return self._heap[0].value

    def add(self, item: T):
        """Add an element to the beam if the size limit has not been reached else pushpop it."""
        # Don't use `__contains__` directly to avoid computing the set key twice
        item_set_key = self.set_key(item)
        if item_set_key in self._set:
            return
        if self.size_limit is None or len(self._set) < self.size_limit:
            heap_item = BeamHeapElem(self.heap_key(item), item_set_key, item)
            heapq.heappush(self._heap, heap_item)
            self._set.add(item_set_key)
        else:
            self.pushpop(item)

    push = add

    def update(self, *others: typing.Iterable[T]):
        for o in others:
            for e in o:
                self.add(e)

    def pop(self) -> T:
        """Return and discard the root of the heap."""
        item = heapq.heappop(self._heap)
        self._set.discard(item.set_key)
        return item.value

    def discard(self, item: T):
        """Discard an element from the beam.

        **Note** since we have to pop an arbitrary element from the heap, this can be very slow.
        """
        item_set_key = self.set_key(item)
        self._set.discard(item_set_key)
        #  This could probably be made more efficient by taking into account the heap structure
        for i, e in enumerate(self._heap):
            if e.set_key == item_set_key:
                self._heap.pop(i)
                heapq.heapify(self._heap)
                return

    def poppush(self, item: T) -> T:
        item_set_key = self.set_key(item)
        heap_item = BeamHeapElem(self.heap_key(item), item_set_key, item)
        old = heapq.heapreplace(self._heap, heap_item)
        self._set.discard(old.set_key)
        self._set.add(item_set_key)
        return old.value

    replace = poppush

    def pushpop(self, item: T) -> T:
        """Push `item` to the beam and pop the minimal element of the result

        Note that it `item` is already present, this doesn't push anything.
        """
        item_set_key = self.set_key(item)
        if item_set_key in self._set:
            return self.pop()
        heap_item = BeamHeapElem(self.heap_key(item), item_set_key, item)
        # Note that `old` here might actually be `item`
        old = heapq.heappushpop(self._heap, heap_item)
        # If `old` was not `item`, swap them in the set
        if hash(old.set_key) != hash(item_set_key):
            self._set.discard(old.set_key)
            self._set.add(item_set_key)
        return old.value

    def copy(self) -> "Beam":
        return

    def __iter__(self) -> typing.Iterator[T]:
        return iter(e.value for e in self._heap)

    def __len__(self) -> int:
        return len(self._set)

    def __contains__(self, item):
        return self._set_key(item) in self._set

    def __repr__(self):
        return f"Beam({list(self)!r}, heap_key={self.heap_key!r}, set_key={self.set_key!r})"


@dataclasses.dataclass
class ValueWrapper:
    """A wrapper to get values from callbacks

    This is slightly slower than using a list, but come on, doesn't it feel better?
    """

    value: typing.Any = None


@torch.jit.script
def unflatten_indice(idx: int, stride: List[int]) -> List[int]:
    res: List[int] = []
    for s in stride:
        dim_idx, idx = divmod(idx, s)
        res.append(dim_idx)
    return res


full_topk_return = namedtuple("full_topk_return", ("values", "indices"))


@torch.jit.script
def full_topk(t: torch.Tensor, k: int = 1, sorted: bool = True) -> full_topk_return:
    flat_topk = t.view(-1).topk(k, sorted=sorted)
    stride: List[int] = []
    # This because bare stride isn't scriptable yet
    for i in range(len(t.shape)):
        stride.append(t.stride(i))
    indices = torch.tensor(
        [
            unflatten_indice(flat_topk.indices[i], stride)
            for i in range(flat_topk.indices.shape[0])
        ],
        dtype=torch.long,
        device=flat_topk.indices.device,
    )
    return full_topk_return(values=flat_topk.values, indices=indices)


def get_embedding_weight(model):
    """
    Extracts and returns the token embedding weight matrix from the model.
    """
    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            for embed in module._token_embedders.keys():
                embedding_weight = module._token_embedders[embed].weight.cpu().detach()
    return embedding_weight


# hook used in add_hooks()
extracted_grads = []


def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])


def observe_output_grad(module: torch.nn.Module):
    wrapper = ValueWrapper()

    def hook(m, grad_in, grad_out):
        wrapper.value = grad_out[0]

    module.register_backward_hook(hook)
    return wrapper


def add_hooks(model):
    """
    Finds the token embedding matrix on the model and registers a hook onto it.
    When loss.backward() is called, extracted_grads list will be filled with
    the gradients w.r.t. the token embeddings
    """
    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            for embed in module._token_embedders.keys():
                module._token_embedders[embed].weight.requires_grad = True
            module.register_backward_hook(extract_grad_hook)


def evaluate_batch(model, batch, trigger_token_ids=None, snli=False):
    """
    Takes a batch of classification examples (SNLI or SST), and runs them through the model.
    If trigger_token_ids is not None, then it will append the tokens to the input.
    This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
    """
    batch = move_to_device(batch[0], cuda_device=0 if torch.cuda.is_available() else -1)
    if trigger_token_ids is None:
        if snli:
            model(batch["premise"], batch["hypothesis"], batch["label"])
        else:
            model(batch["tokens"], batch["label"])
        return None
    else:
        trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
        trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch["label"]), 1)
        if torch.cuda.is_available():
            trigger_sequence_tensor = trigger_sequence_tensor.cuda()
        if snli:
            original_tokens = batch["hypothesis"]["tokens"].clone()
            batch["hypothesis"]["tokens"] = torch.cat(
                (trigger_sequence_tensor, original_tokens), 1
            )
            output_dict = model(batch["premise"], batch["hypothesis"], batch["label"])
            batch["hypothesis"]["tokens"] = original_tokens
        else:
            original_tokens = batch["tokens"]["tokens"].clone()
            batch["tokens"]["tokens"] = torch.cat(
                (trigger_sequence_tensor, original_tokens), 1
            )
            output_dict = model(batch["tokens"], batch["label"])
            batch["tokens"]["tokens"] = original_tokens
        return output_dict


def get_average_grad(model, batch, trigger_token_ids, target_label=None, snli=False):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.
    """
    # create an dummy optimizer for backprop
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    # prepend triggers to the batch
    original_labels = batch[0]["label"].clone()
    if target_label is not None:
        # set the labels equal to the target (backprop from the target class, not model prediction)
        batch[0]["label"] = int(target_label) * torch.ones_like(batch[0]["label"])
        if torch.cuda.is_available:
            batch[0]["label"] = batch[0]["label"].cuda()
    global extracted_grads
    extracted_grads = []  # clear existing stored grads
    loss = evaluate_batch(model, batch, trigger_token_ids, snli)["loss"]
    loss.backward()
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads = extracted_grads[0].cpu()
    batch[0]["label"] = original_labels  # reset labels

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad = averaged_grad[
        0 : len(trigger_token_ids)
    ]  # return just trigger grads
    return averaged_grad


def get_accuracy(model, dev_dataset, vocab, trigger_token_ids=None, snli=False):
    """
    When trigger_token_ids is None, gets accuracy on the dev_dataset. Otherwise, gets accuracy with
    triggers prepended for the whole dev_dataset.
    """
    model.get_metrics(reset=True)
    model.eval()  # model should be in eval() already, but just in case
    if snli:
        iterator = BucketIterator(
            batch_size=128, sorting_keys=[("premise", "num_tokens")]
        )
    else:
        iterator = BucketIterator(
            batch_size=128, sorting_keys=[("tokens", "num_tokens")]
        )
    iterator.index_with(vocab)
    if trigger_token_ids is None:
        for batch in lazy_groups_of(
            iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1
        ):
            evaluate_batch(model, batch, trigger_token_ids, snli)
        print("Without Triggers: " + str(model.get_metrics()["accuracy"]))
    else:
        print_string = ""
        for idx in trigger_token_ids:
            print_string = print_string + vocab.get_token_from_index(idx) + ", "

        for batch in lazy_groups_of(
            iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1
        ):
            evaluate_batch(model, batch, trigger_token_ids, snli)
        print(
            "Current Triggers: "
            + print_string
            + " : "
            + str(model.get_metrics()["accuracy"])
        )


def get_best_candidates(
    model, batch, trigger_token_ids, cand_trigger_token_ids, snli=False, beam_size=1
):
    """
    Given the list of candidate trigger token ids (of number of trigger words by number of
    candidates per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(
        0, model, batch, trigger_token_ids, cand_trigger_token_ids, snli
    )
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(
        1, len(trigger_token_ids)
    ):  # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for (
            cand,
            _,
        ) in top_candidates:  # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(
                get_loss_per_candidate(
                    idx, model, batch, cand, cand_trigger_token_ids, snli
                )
            )
        top_candidates = heapq.nlargest(
            beam_size, loss_per_candidate, key=itemgetter(1)
        )
    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate(
    index, model, batch, trigger_token_ids, cand_trigger_token_ids, snli=False
):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    model.get_metrics(reset=True)
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    curr_loss = (
        evaluate_batch(model, batch, trigger_token_ids, snli)["loss"]
        .cpu()
        .detach()
        .numpy()
    )
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids)  # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][
            cand_id
        ]  # replace one token
        loss = (
            evaluate_batch(model, batch, trigger_token_ids_one_replaced, snli)["loss"]
            .cpu()
            .detach()
            .numpy()
        )
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate
