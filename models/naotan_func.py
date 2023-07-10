
import torch


def sp_new_values(t, values):
    out = torch.sparse_coo_tensor(t._indices(), values, t.shape)
    # If the input tensor was coalesced, the output one will be as well since we don't modify the indices.
    if t.is_coalesced():
        with torch.no_grad():
            out._coalesced_(True)
    return out


def pairwise_cosine(X):
    pairwise_feat_dot_prods = X @ X.transpose(-2, -1)  # pfdp_ij = <X_i|X_j>
    range_ = torch.arange(pairwise_feat_dot_prods.shape[-1])
    feat_norms = pairwise_feat_dot_prods[..., range_, range_].sqrt()  # fn_i = ||X_i||_2
    feat_norms = torch.where(feat_norms < 1e-8, torch.tensor(1.0, device=X.device), feat_norms)
    return pairwise_feat_dot_prods / feat_norms[..., :, None] / feat_norms[..., None, :]



def Sum(t, dim, dense: bool = False):
    return (torch.sparse.sum(t, dim).to_dense() if dense else torch.sparse.sum(t, dim)) if t.is_sparse else t.sum(dim)

def Neq0(t):
    return _operate_on_nonzero_elems(lambda elems: elems != 0, t, op_requires_coalesced=True, op_keeps_0=True)

def _operate_on_nonzero_elems(
        op,
        t,
        op_requires_coalesced: bool,
        op_keeps_0: bool
):
    if t.is_sparse:
        if op_requires_coalesced and not t.is_coalesced():
            raise ValueError(f"Sparse tensor must be coalesced for applying the element-wise operation {op}")
        return sp_new_values(t, _operate_on_nonzero_elems(op, t._values(), False, op_keeps_0))
    else:
        if op_keeps_0:
            return op(t)
        else:
            tr = t.clone()
            idx = tuple(tr.nonzero().T)
            tr[idx] = op(tr[idx])
            return tr


def Mul(t1, t2):
    return _combine_elem_wise(torch.mul, t1, t2, requires_coalesced=False)



def _combine_elem_wise(
        op,
        t1,
        t2,
        requires_coalesced: bool
) :
    # t1_sp = t1.is_sparse
    # t2_sp = t2.is_sparse
    # if t1_sp and not t2_sp:
    #     if requires_coalesced and not t1.is_coalesced():
    #         raise ValueError(f"Sparse tensor must be coalesced for applying the element-wise operation {op}")
    #     t2_values = t2.broadcast_to(t1.shape)[tuple(t1._indices())]
    #     return sp_new_values(t1, op(t1._values(), t2_values))
    # elif not t1_sp and t2_sp:
    #     if requires_coalesced and not t2.is_coalesced():
    #         raise ValueError(f"Sparse tensor must be coalesced for applying the element-wise operation {op}")
    #     t1_values = t1.broadcast_to(t2.shape)[tuple(t2._indices())]
    #     return sp_new_values(t2, op(t1_values, t2._values()))
    # else:
    #     return op(t1, t2)

    return op(t1, t2)


def Sp_diag(values):
    shape = (*values.shape, values.shape[-1])
    if values.ndim == 1:
        return torch.sparse_coo_tensor(
            indices=torch.arange(values.shape[0], device=values.device).tile(2, 1),
            values=values,
            size=shape
        )
    else:
        return torch.sparse_coo_tensor(
            indices=torch.vstack([
                torch.arange(values.shape[0], device=values.device).repeat_interleave(values.shape[1]),
                torch.arange(values.shape[1], device=values.device).tile(2, values.shape[0])
            ]),
            values=values.flatten(),
            size=shape
        )