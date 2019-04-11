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
@click.option('-k', '--kpoints-distance', type=click.FLOAT, required=True, default=0.15)
@click.option(
    '-f', '--final-scf', is_flag=True, default=False, show_default=True,
    help='run a final scf calculation for the final relaxed structure'
)
@click.option(
    '-g', '--group', type=click.STRING, required=False,
    help='the label of a Group to add the final PwCalculation to in case of success'
)
def launch(
    code, structure, pseudo_family, kpoints, max_num_machines, max_wallclock_seconds, daemon, ecutwfc, ecutrho,
    hubbard_u, hubbard_v, hubbard_file_pk, starting_magnetization, smearing, automatic_parallelization, clean_workdir,
    final_scf, group, kpoints_distance):
    """
    Run the PwRelaxWorkChain for a given input structure
    """
    from aiida.orm.data.base import Bool, Float, Str
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.utils import WorkflowFactory
    from aiida.work.launch import run, submit
    from aiida_quantumespresso.utils.resources import get_default_options, get_automatic_parallelization_options

    PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
    
    # ============================================= #
    nnodes = 1
    nthreads = 3  # ignored for normal QE (nthreads=1)
    nranks_per_node = 4
    kpools = nranks_per_node
    # ============================================= #

    parameters = {
        'SYSTEM': {
            'ecutwfc': ecutwfc,
            'ecutrho': ecutrho,
        },
        'CONTROL': {
            #'wf_collect': False,  # WARNING: needed for SIRIUS
            'tprnfor': True,
            'tstress': True,
            'forc_conv_thr': 0.0001,
        },
        'ELECTRONS': {
            #'startingwfc': 'file',  # WARNING: QE-GPU doesn't seem to like this option
            'conv_thr': 1e-9 * len(structure.sites),  # TODO: change me back too 1e-10*len(sites) when running larger structures!!
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

    inputs = {
        'structure': structure,
        'base': {
            'code': code,
            'pseudo_family': Str(pseudo_family),
            'kpoints_distance': Float(kpoints_distance),
            'parameters': ParameterData(dict=parameters),
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
    inputs['base']['settings'] = settings

    if automatic_parallelization:
        automatic_parallelization = get_automatic_parallelization_options(max_num_machines, max_wallclock_seconds)
        inputs['base']['automatic_parallelization'] = ParameterData(dict=automatic_parallelization)
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
        inputs['base']['options'] = options
    #else:
    #    options = get_default_options(max_num_machines, max_wallclock_seconds)
    #    inputs['base']['options'] = ParameterData(dict=options)

    if clean_workdir:
        inputs['clean_workdir'] = Bool(True)

    if final_scf:
        inputs['final_scf'] = Bool(True)

    if group:
        inputs['group'] = Str(group)

    if daemon:
        workchain = submit(PwRelaxWorkChain, **inputs)
        click.echo('Submitted {}<{}> to the daemon'.format(PwRelaxWorkChain.__name__, workchain.pk))
    else:
        run(PwRelaxWorkChain, **inputs)
