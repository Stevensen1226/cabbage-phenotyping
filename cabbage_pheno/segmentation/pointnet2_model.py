import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes, additional_channel=0):
        super(PointNet2SemSeg, self).__init__()
        # SA1: Input (3 + additional_channel) -> Output 64
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3 + additional_channel, [32, 32, 64], False)
        # SA2: Input 64 + 3 -> Output 128
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        # SA3: Input 128 + 3 -> Output 256
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        # SA4: Input 256 + 3 -> Output 512
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        
        # FP4: Input 512 + 256 -> Output 256
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        # FP3: Input 256 + 128 -> Output 256
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        # FP2: Input 256 + 64 -> Output 128
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        
        # FP1: Input 128 + (3+additional)
        # xyz1 (original), xyz2 (sa1_xyz)
        # points1 (original features), points2 (fp2 output)
        self.fp1 = PointNetFeaturePropagation(128 + 3 + additional_channel, [128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, features=None):
        """
        Input:
            xyz: [B, 3, N]
            features: [B, C, N] (optional additional features like color/intensity)
        """
        l0_xyz = xyz
        # For SA1, we only pass features (if any). 
        # SA1 will calculate relative coordinates automatically.
        # We do NOT pass xyz as features here to avoid channel mismatch (3 vs 6).
        l0_points = features

        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature Propagation layers
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # For FP1, we DO want to concatenate XYZ with features as the "original points"
        l0_points_with_xyz = torch.cat([l0_xyz, features], 1) if features is not None else l0_xyz
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points_with_xyz, l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x, l0_points
