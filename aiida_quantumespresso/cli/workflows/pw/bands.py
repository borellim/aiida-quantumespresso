# -*- coding: utf-8 -*-
import click
from aiida.utils.cli import command
from aiida.utils.cli import options
from aiida_quantumespresso.utils.cli import options as options_qe
from aiida_quantumespresso.utils.cli import validate

def get_cutoffs_from_stored_dict(structure, pseudo_family):
    """
    Retrieve cutoff / dual information from a stored node
    """
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.querybuilder import QueryBuilder

    filters = {
        'attributes.pseudo_family_name': pseudo_family,
    }

    builder = QueryBuilder().append(ParameterData, filters=filters)

    if builder.count() > 1:
        raise ValueError('multiple ParameterData nodes with cutoffs found for {} pseudo family'.format(pseudo_family))
    elif builder.count() == 0:
        raise ValueError('no ParameterData nodes with cutoffs found for {} pseudo family'.format(pseudo_family))
    else:
        param_node = builder.all()[0][0]
    
    assert(param_node.uuid == "fc69a15f-95fa-425b-a267-88ed3b748c3d")  # hardcoded check for SSSP acc PBE v1.0 (new aiida_dev installation)

    cutoffs_dict = param_node.get_dict()['elements']
    symbols_set = structure.get_symbols_set()
    cutoffs_wfc = [float(cutoffs_dict[_]['cutoff']) for _ in symbols_set]
    cutoffs_rho = [float(cutoffs_dict[_]['cutoff']) * float(cutoffs_dict[_]['dual']) for _ in symbols_set]

    # take the maximum cutoff for all species
    cutoff_wfc = max(cutoffs_wfc)
    cutoff_rho = max(cutoffs_rho)
    return (cutoff_wfc, cutoff_rho) 

#def get_cutoffs_from_stored_dict(structure, pseudo_family):
#    """
#    Retrieve cutoff / dual information from a stored node
#    NOTE: we load directly from a hardcoded UUID
#    """
#    #uuid = "ff62a41b-8db7-446c-9cb7-e50474d783af"  # SSSP eff PBE v1.0
#    #uuid = "ad39bf0c-9e46-41ea-afb4-0d06d747b479"  # SSSP acc PBE v1.0 (old aiida installation)
#    uuid = "fc69a15f-95fa-425b-a267-88ed3b748c3d"  # SSSP acc PBE v1.0 (new aiida_dev installation)
#    data = load_node(uuid=uuid).get_attrs()
#    assert(data['pseudo_family_name'] == pseudo_family)
#    cutoffs_dict = data['elements']
#    symbols_set = structure.get_symbols_set()
#    assert(all([cutoffs_dict[_]['cutoff_units'] == 'Ry' for _ in symbols_set]))
#    cutoffs_wfc = [float(cutoffs_dict[_]['cutoff']) for _ in symbols_set]
#    cutoffs_rho = [float(cutoffs_dict[_]['cutoff'])*
#                    float(cutoffs_dict[_]['dual']) for _ in symbols_set]
#    # take the maximum cutoff for all species
#    cutoff_wfc = np.max(cutoffs_wfc)
#    cutoff_rho = np.max(cutoffs_rho)
#    return (cutoff_wfc, cutoff_rho)


@command()
@options.code(callback_kwargs={'entry_point': 'quantumespresso.pw'})
@options.structure()
@options.pseudo_family()
@options.kpoint_mesh()
@options.max_num_machines()
@options.max_wallclock_seconds()
@options.daemon()
#@options_qe.ecutwfc()
#@options_qe.ecutrho()
@options_qe.hubbard_u()
@options_qe.hubbard_v()
@options_qe.hubbard_file()
@options_qe.starting_magnetization()
@options_qe.smearing()
@options_qe.automatic_parallelization()
@options_qe.clean_workdir()
def launch(
    code, structure, pseudo_family, kpoints, max_num_machines, max_wallclock_seconds, daemon, #ecutwfc, ecutrho,
    hubbard_u, hubbard_v, hubbard_file_pk, starting_magnetization, smearing, automatic_parallelization, clean_workdir):
    """
    Run the PwBandsWorkChain for a given input structure
    """
    from aiida.orm.data.base import Bool, Float, Str
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.utils import WorkflowFactory
    from aiida.work.launch import run, submit
    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options

    PwBandsWorkChain = WorkflowFactory('quantumespresso.pw.bands')

    # ============================================= #
    nnodes = 1
    nthreads = 2  # ignored for normal QE (nthreads=1)
    nranks_per_node = 6
    kpools = nranks_per_node
    # ============================================= #

    cutoff_wfc, cutoff_rho = get_cutoffs_from_stored_dict(structure, pseudo_family)

    parameters = {
        'SYSTEM': {
            'ecutwfc': cutoff_wfc,
            'ecutrho': cutoff_rho,
        },
        'CONTROL': {
            #'wf_collect': False,  # WARNING: 'false' needed for SIRIUS
            'tprnfor': True,
            'tstress': True,
            'forc_conv_thr': 0.0001,  # Nico?
        },
        'ELECTRONS': {
            #'startingwfc': 'file',  # WARNING: QE-GPU doesn't seem to like this option
            'conv_thr': 2e-9 * len(structure.sites),  # WARNING: changed for testing - was 2e-10*len(sites)
        },
    }

    try:
        hubbard_file = validate.validate_hubbard_parameters(structure, parameters, hubbard_u, hubbard_v, hubbard_file_pk)
    except ValueError as exception:
        raise click.BadParameter(exception.message)

    try:
        validate.validate_starting_magnetization(structure, parameters, starting_magnetization)
    except ValueError as exception:
        raise click.BadParameter(exception.message)

    try:
        validate.validate_smearing(parameters, smearing)
    except ValueError as exception:
        raise click.BadParameter(exception.message)


    pseudo_family = Str(pseudo_family)
    parameters = ParameterData(dict=parameters)

    inputs = {
        'structure': structure,
        'relax': {
            'base': {
                'code': code,
                'pseudo_family': pseudo_family,
                'kpoints_distance': Float(0.4),  # 0.15 ?
                'parameters': parameters,
            },
            'meta_convergence': Bool(True),
        },
        'scf': {
            'code': code,
            'pseudo_family': pseudo_family,
            'kpoints_distance': Float(0.15),
            'parameters': parameters,
        },
        'bands': {
            'code': code,
            'pseudo_family': pseudo_family,
            'kpoints_distance': Float(0.15),
            'parameters': parameters,
        }
    }
    
    if 'sirius' in code.label.lower():
        settings = {
            'cmdline': ['-nk', str(kpools), '-sirius', '-sirius_cfg', '/users/mborelli/sirius_cfg/sirius_config_anton_20180329.json']
        }
    else:
        settings = {
            'cmdline': ['-nk', str(kpools)]
        }
    settings = ParameterData(dict=settings)
    inputs['relax']['base']['settings'] = settings
    inputs['scf']['settings'] = settings
    inputs['bands']['settings'] = settings

    if automatic_parallelization:
        auto_para = ParameterData(dict=get_automatic_parallelization_options(max_num_machines, max_wallclock_seconds))
        inputs['relax']['base']['automatic_parallelization'] = auto_para
        inputs['scf']['automatic_parallelization'] = auto_para
        inputs['bands']['automatic_parallelization'] = auto_para
    #elif 'sirius' in code.label.lower():
    else:
        options = ParameterData(dict={
            'resources': {
                'num_machines': nnodes,
                'num_mpiprocs_per_machine': nranks_per_node,
                'num_cores_per_mpiproc': nthreads,
            },
            'max_wallclock_seconds': 60*60*2
        })
        inputs['relax']['base']['options'] = options
        inputs['scf']['options'] = options
        inputs['bands']['options'] = options
    #else:
    #    options = ParameterData(dict=get_default_options(max_num_machines, max_wallclock_seconds))
    #    inputs['relax']['base']['options'] = options
    #    inputs['scf']['options'] = options
    #    inputs['bands']['options'] = options

    if clean_workdir:
        inputs['clean_workdir'] = Bool(True)

    if daemon:
        workchain = submit(PwBandsWorkChain, **inputs)
        click.echo('Submitted {}<{}> to the daemon'.format(PwBandsWorkChain.__name__, workchain.pk))
    else:
        run(PwBandsWorkChain, **inputs)
