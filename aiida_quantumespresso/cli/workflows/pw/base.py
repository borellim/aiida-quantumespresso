# -*- coding: utf-8 -*-
import click
from aiida.utils.cli import command
from aiida.utils.cli import options
from aiida_quantumespresso.utils.cli import options as options_qe
from aiida_quantumespresso.utils.cli import validate


@command()
@options.code(callback_kwargs={'entry_point': 'quantumespresso.pw'})
@options.structure()
@options.pseudo_family()
@options.kpoint_mesh()
@options.max_num_machines()
@options.max_wallclock_seconds()
@options.daemon()
@options_qe.ecutwfc()
@options_qe.ecutrho()
@options_qe.hubbard_u()
@options_qe.hubbard_v()
@options_qe.hubbard_file()
@options_qe.starting_magnetization()
@options_qe.smearing()
@options_qe.automatic_parallelization()
@options_qe.clean_workdir()
def launch(
    code, structure, pseudo_family, kpoints, max_num_machines, max_wallclock_seconds, daemon, ecutwfc, ecutrho,
    hubbard_u, hubbard_v, hubbard_file_pk, starting_magnetization, smearing, automatic_parallelization, clean_workdir):
    """
    Run the PwBaseWorkChain for a given input structure
    """
    from aiida.orm.data.base import Bool, Str, Float
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.utils import WorkflowFactory
    from aiida.work.launch import run, submit
    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options
    from aiida.orm.utils import load_node   # WARNING! hack to test restart

    PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')

    # ============================================= #
    nnodes = 1
    nthreads = 3  # ignored for normal QE (nthreads=1)
    nranks_per_node = 4
    kpools = nranks_per_node
    # ============================================= #

    parameters = {
        'CONTROL': {
            'calculation': 'nscf', # 'nscf',  # WARNING: hack
            'restart_mode': 'restart',  # WARNING! hack to test restart
            #'wf_collect': False,   # WARNING: needed by SIRIUS
        },
        'SYSTEM': {
            'ecutwfc': ecutwfc,
            'ecutrho': ecutrho,
        },
        'ELECTRONS': {
            #'startingwfc': 'file',  # WARNING: QE-GPU doesn't seem to like this option
            'conv_thr': 1e-9 * len(structure.sites)
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

    parent_calculation = load_node(3153)  # 2379 # 2430      # 2067 # 2216
    parent_folder = parent_calculation.out.remote_folder
    inputs = {
        'code': code,
        'structure': structure,
        'parent_folder': parent_folder,  # WARNING! hack to test restart
        'pseudo_family': Str(pseudo_family),
        'kpoints': kpoints,  # WARNING: overrides kpoints_distance
        #'kpoints_distance': Float(0.2),  # WARNING: changed to test QE-GPU and SIRIUS -- overridden by 'kpoints'
        'parameters': ParameterData(dict=parameters),
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
    inputs['settings'] = settings

    if automatic_parallelization:
        automatic_parallelization = get_automatic_parallelization_options(max_num_machines, max_wallclock_seconds)
        inputs['automatic_parallelization'] = ParameterData(dict=automatic_parallelization)
    #elif 'sirius' in code.label.lower():
    else:
        options = ParameterData(dict={
            'resources': {
                'num_machines': nnodes,
                'num_mpiprocs_per_machine': nranks_per_node,
                'num_cores_per_mpiproc': nthreads,
            },
            'max_wallclock_seconds': 60*30
        })
        inputs['options'] = options
    #else:
    #    options = get_default_options(max_num_machines, max_wallclock_seconds)
    #    inputs['options'] = ParameterData(dict=options)

    if clean_workdir:
        inputs['clean_workdir'] = Bool(True)

    if daemon:
        workchain = submit(PwBaseWorkChain, **inputs)
        click.echo('Submitted {}<{}> to the daemon'.format(PwBaseWorkChain.__name__, workchain.pk))
    else:
        run(PwBaseWorkChain, **inputs)
