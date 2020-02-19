from fireworks import LaunchPad, Firework
from mldftdat.pyscf_tasks import SCFCalcConvergenceFixer, SCFCalc

lpad = LaunchPad.auto_load()

fw_ids = lpad.get_fw_ids(query = {'state' : 'FIZZLED'})
new_fws = []
unknown_fail_reasons = []
num_fws_added = 0
MAX_NUM_FWS_ADDED = 100

for fw_id in fw_ids:
    fw = lpad.get_fw_by_id(fw_id)
    task = fw.tasks[0]
    if fw.launches[-1].action is None:
        exception = 'MEMORY ERROR'
    else:
        exception = fw.launches[-1].action.stored_data['_exception']['_stacktrace']
    if isinstance(task, SCFCalc) and\
            ('did not converge' in exception)\
            and ('CCSD' in exception):
        print(fw_id, 'did not converge CCSD')
        task.pop('_fw_name')
        new_task = SCFCalcConvergenceFixer(**task)
        new_fw = Firework([new_task] + fw.tasks[1:], name=fw.name + '_CONVFIX')
        if num_fws_added < MAX_NUM_FWS_ADDED:
            print('ADDING NEW FW', fw_id)
            num_fws_added += 1
            lpad.defuse_fw(fw_id)
            lpad.add_wf(new_fw)
    elif isinstance(task, SCFCalcConvergenceFixer) and\
            ('did not converge' in exception)\
            and ('SCF' in exception): # condition no in original script
        print ('Defusing fw', fw_id)
        lpad.defuse_fw(fw_id)
        #lpad.rerun_fw(fw_id)
    elif exception == 'MEMORY ERROR':
        print(fw_id, 'ran out of memory')
    elif isinstance(task, SCFCalc) and\
            ('Basis not found' in exception):
        print(fw_id, 'did not have a basis set for the structure')
        lpad.archive_wf(fw_id)
    else:
        print(fw_id, 'had an unknown error')
        unknown_fail_reasons.append(fw_id)

print('The following fw_ids failed for unknown reasons:')
print(unknown_fail_reasons)
print('Added this many new fws')
print(num_fws_added)
