import pytest
import torch

from openmixup.models.necks import (GeneralizedMeanPooling, AvgPoolNeck, LinearNeck)

def test_gap_neck():

    # test 1d gap_neck
    neck = AvgPoolNeck(dim=1)
    # batch_size, num_features, feature_size
    fake_input = torch.rand(1, 16, 24)

    output = neck(fake_input)
    # batch_size, num_features
    assert output[0].shape == torch.Size([16, 1])

    # test 1d gap_neck
    neck = AvgPoolNeck(dim=2)
    # batch_size, num_features, feature_size(2)
    fake_input = torch.rand(1, 16, 24, 24)

    output = neck(fake_input)
    # batch_size, num_features
    assert output[0].shape == torch.Size([16, 1, 1])

    # test 1d gap_neck
    neck = AvgPoolNeck(dim=3)
    # batch_size, num_features, feature_size(3)
    fake_input = torch.rand(1, 16, 24, 24, 5)

    output = neck(fake_input)
    # batch_size, num_features
    assert output[0].shape == torch.Size([16, 1, 1, 1])

    with pytest.raises(AssertionError):
        # dim must in [1, 2, 3]
        AvgPoolNeck(dim='other')


def test_gem_neck():

    # test gem_neck
    neck = GeneralizedMeanPooling()
    # batch_size, num_features, feature_size(2)
    fake_input = torch.rand(1, 16, 24, 24)

    output = neck(fake_input)
    # batch_size, num_features
    assert output[0].shape == torch.Size([1, 16])


    with pytest.raises(AssertionError):
        # p must be a value greater then 1
        GeneralizedMeanPooling(p=0.5)
        
        
def test_linear_neck():

    # test linear neck
    neck = LinearNeck(in_channels=1, out_channels=1)
    # batch_size, num_features, feature_size
    fake_input = torch.rand(1, 16, 24, 24)

    output = neck(fake_input)
    # batch_size, num_features
    assert output[0].shape == torch.Size([16, 1])


