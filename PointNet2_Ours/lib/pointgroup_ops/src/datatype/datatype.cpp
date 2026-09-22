#include "datatype.h"

template <Int dimension> SparseGrid<dimension>::SparseGrid() : ctr(0) {}

ConnectedComponent::ConnectedComponent() {}

void ConnectedComponent::addPoint(Int pt_idx) {
    pt_idxs.push_back(pt_idx);
}
