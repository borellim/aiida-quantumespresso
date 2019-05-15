# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Tests for the `PwParser`."""
from __future__ import absolute_import

import pytest
from aiida import orm
from aiida.common import AttributeDict


@pytest.fixture
def generate_inputs():
    """Return only those inputs that the parser will expect to be there."""
    structure = orm.StructureData()
    parameters = {
        'CONTROL': {
            'calculation': 'scf'
        },
        'SYSTEM': {
            'ecutrho': 240.0,
            'ecutwfc': 30.0
        }
    }
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(0.15)

    return AttributeDict({
        'structure': structure,
        'kpoints': kpoints,
        'parameters': orm.Dict(dict=parameters),
        'settings': orm.Dict()
    })


@pytest.fixture
def generate_inputs_vcrelax_Ge2():
    """Return only those inputs that the parser will expect to be there."""
    structure = orm.StructureData()
    structure.set_cell(
        [[0.0, 2.87760769034995, 2.87760769034995],
         [2.87760769034995, 0.0, 2.87760769034995],
         [2.87760769034995, 2.87760769034995, 0.0]]
    )
    structure.append_atom(name='Ge1', symbols='Ge', position=[0.]*3)
    structure.append_atom(name='Ge2', symbols='Ge', position=[1.43880384517]*3)
    parameters = {
        'CONTROL': {
            'calculation': 'vc-relax'
        },
        'SYSTEM': {
            'ecutrho': 240.0,
            'ecutwfc': 30.0
        }
    }
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh([2,2,2],[0.,0.,0.])

    return AttributeDict({
        'structure': structure,
        'kpoints': kpoints,
        'parameters': orm.Dict(dict=parameters),
        'settings': orm.Dict()
    })

# @pytest.fixture
# def generate_inputs_vcrelax_Ge2_deprkeys():
#     inputs = generate_inputs_vcrelax_Ge2()
#     inputs.settings.include_deprecated_v2_keys = True
#     return inputs


def test_pw_default(fixture_database, fixture_computer_localhost, generate_calc_job_node, generate_parser,
                    generate_inputs, data_regression):
    """Test a default `pw.x` calculation.

    The output is created by running a dead simple SCF calculation for a silicon structure.
    This test should test the standard parsing of the stdout content and XML file stored in the standard results node.
    """
    entry_point_calc_job = 'quantumespresso.pw'
    entry_point_parser = 'quantumespresso.pw'

    node = generate_calc_job_node(entry_point_calc_job, fixture_computer_localhost, 'default', generate_inputs)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_parameters' in results
    data_regression.check(results['output_parameters'].get_dict())


def test_pw_vcrelax_Ge2_qe63_oldxml(fixture_database, fixture_computer_localhost, generate_calc_job_node,
                                    generate_parser, generate_inputs_vcrelax_Ge2, data_regression):
    """Test a default `pw.x` calculation.

    The output is created by running a dead simple SCF calculation for a silicon structure.
    This test should test the standard parsing of the stdout content and XML file stored in the standard results node.
    """
    entry_point_calc_job = 'quantumespresso.pw'
    entry_point_parser = 'quantumespresso.pw'

    node = generate_calc_job_node(entry_point_calc_job, fixture_computer_localhost, 'vcrelax_Ge2_qe63_oldxml',
                                  generate_inputs_vcrelax_Ge2)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_parameters' in results
    data_regression.check(
        results['output_parameters'].get_dict(),
    )


def test_pw_vcrelax_Ge2_qe63_newxml(fixture_database, fixture_computer_localhost, generate_calc_job_node,
                                    generate_parser, generate_inputs_vcrelax_Ge2, data_regression):
    """Test a default `pw.x` calculation.

    The output is created by running a dead simple SCF calculation for a silicon structure.
    This test should test the standard parsing of the stdout content and XML file stored in the standard results node.
    """
    entry_point_calc_job = 'quantumespresso.pw'
    entry_point_parser = 'quantumespresso.pw'

    node = generate_calc_job_node(entry_point_calc_job, fixture_computer_localhost, 'vcrelax_Ge2_qe63_newxml',
                                  generate_inputs_vcrelax_Ge2)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_parameters' in results
    data_regression.check(
        results['output_parameters'].get_dict(),
    )

def test_pw_vcrelax_Ge2_qe63_newxml_deprkeys(fixture_database, fixture_computer_localhost, generate_calc_job_node,
                                             generate_parser, generate_inputs_vcrelax_Ge2, data_regression):
    """Test the new XML parser with 'include_deprecated_v2_keys' option, versus without."""
    entry_point_calc_job = 'quantumespresso.pw'
    entry_point_parser = 'quantumespresso.pw'

    inputs = generate_inputs_vcrelax_Ge2
    inputs.settings.update_dict({'parser_options': {'include_deprecated_v2_keys': True}})
    node = generate_calc_job_node(entry_point_calc_job, fixture_computer_localhost, 'vcrelax_Ge2_qe63_newxml', inputs)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_parameters' in results
    
    # Remove and check the extra deprecated keys
    out_params = results['output_parameters'].get_dict()
    assert out_params.pop('fixed_occupations') == False
    assert out_params.pop('smearing_method') == True
    assert out_params.pop('tetrahedron_method') == False
    
    data_regression.check(
        out_params,
        basename='test_pw_vcrelax_Ge2_qe63_newxml'
    )
    
# TODO:
#  - check other outputs (structure, bands, kpoints, trajectory, ...)
#   (run this on the aiida_0.12 environment: meld <(verdi run print_results.py 4015) <(verdi run print_results.py 441)
#  - check old aiida-qe 2.1 vs 3.0
#  - explicitly check the differences between old XML and new XML