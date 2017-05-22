import yaml
import os
import shutil
from sqlalchemy import Table, MetaData, Column, Integer, Float, String, create_engine

GAS_RAW_DATA_ROOT = os.path.join('pytpc', 'data', 'raw')
GAS_LIST_PATH = os.path.join(GAS_RAW_DATA_ROOT, 'gas_list.yml')
GAS_DB_PATH = os.path.join('pytpc', 'data', 'gases', 'gasdata.db')


def read_srim(fp):
    en_units = {'eV': 1e-6, 'keV': 1e-3, 'MeV': 1, 'GeV': 1e3}
    dist_units = {'um': 1e-4, 'mm': 1e-1, 'cm': 1, 'm': 1e2}

    res = []

    with open(fp) as f:
        for line in f:
            if 'Density' in line:
                litems = line.strip().split()
                gas_dens = float(litems[3])
                assert litems[4] == 'g/cm3', 'Units for density are not g/cm3: {}'.format(litems[4])
            if 'Straggling' in line:
                f.readline()
                break
        for line in f:
            if '-------' in line:
                break
            litems = line.strip().split()
            if len(litems) != 10:
                raise ValueError('Wrong number of entries in line!')

            en = float(litems[0]) * en_units[litems[1]]
            dedx_elec = float(litems[2]) * 1000  # convert MeV/(mg/cm^2) to MeV/(g/cm^2)
            dedx_nuc = float(litems[3]) * 1000
            proj_range = float(litems[4]) * dist_units[litems[5]] * gas_dens

            res.append({'energy': en, 'dedx': dedx_elec + dedx_nuc, 'range': proj_range})

    return res


def read_astar(fp):
    result = []
    with open(fp, 'r') as f:
        # Skip the header
        for i in range(7):
            f.readline()

        for line in f:
            energy, dedx, range_ = (float(x) for x in line.strip().split(' '))
            result.append({'energy': energy, 'dedx': dedx, 'range': range_})

    return result


def read_file(path, source):
    if source == 'srim':
        return read_srim(path)
    elif source == 'astar':
        return read_astar(path)
    else:
        raise ValueError('Unknown data source {}'.format(source))


def prepare_entry_for_particle(data_path, data_source):
    data = read_file(data_path, data_source)
    return data


def load_gas_list(path):
    with open(path, 'r') as f:
        return yaml.load(f)


def build():
    try:
        os.remove(GAS_DB_PATH)
    except FileNotFoundError:
        pass

    engine = create_engine('sqlite:///{}'.format(GAS_DB_PATH))
    gas_list = load_gas_list(GAS_LIST_PATH)
    meta = MetaData(engine)

    # Build mass table
    mass_table = Table(
        'masses', meta,
        Column('name', String, primary_key=True),
        Column('mass', Float),
    )
    mass_table.create()

    for gas_name, gas_props in gas_list.items():
        gas_table = Table(
            gas_name, meta,
            Column('energy', Float),
            Column('dedx', Float),
            Column('range', Float),
            Column('proj_mass', Integer),
            Column('proj_charge', Integer),
        )
        gas_table.create()

        mass_ins = mass_table.insert().values(name=gas_name, mass=float(gas_props['mass']))
        with engine.begin() as conn:
            conn.execute(mass_ins)

        for particle in gas_props['particles']:
            data = prepare_entry_for_particle(
                data_path=os.path.join(GAS_RAW_DATA_ROOT, particle['file']),
                data_source=particle['source']
            )
            data_ins = gas_table.insert().values(proj_mass=int(particle['mass']), proj_charge=int(particle['charge']))

            with engine.begin() as conn:
                conn.execute(data_ins, data)


if __name__ == '__main__':
    build()
