"""Bind classifier cost types to actual frozen training and evaluation members."""
import hashlib,json

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def members(value):
    rows=value['positions'] if isinstance(value,dict) else value
    if any(isinstance(x,bool) or not isinstance(x,int) and type(x).__name__ not in ('int32','int64') for x in rows):raise ValueError('Noninteger membership')
    result=[int(x) for x in rows]
    if not result or len(result)!=len(set(result)):raise ValueError('Empty or duplicate membership')
    return result

class MembershipBinding:
    def __init__(self,partitions,tasks,features):
        self.parts=partitions;self.tasks={t['task_id']:t for t in tasks};self.features=tuple(features)
        self.outers={o['fold']:o for o in partitions['outer_folds']}
        self.outer_sets={f:{k:frozenset(members(o[k])) for k in ('pool','original_train','test')} for f,o in self.outers.items()}
        self.context_sets={key:{k:frozenset(members(c[k])) for k in ('train','validation') if k in c} for key,c in partitions['contexts'].items()}
        self.games={(key,g['resample_index']):g for key,c in partitions['contexts'].items() for g in c['games']}
        self.utility_sets={(key,r,f['fold']):(frozenset(f['train_positions']),frozenset(f['validation_positions'])) for (key,r),g in self.games.items() for f in g['utility_folds']}
        self.mask_sets={(key,r,m):frozenset(g['nonempty_coalition_masks'][m]) for (key,r),g in self.games.items() for m in ('full','robust_half')}
        self.task_predictions={key:frozenset(t['prediction_positions']) for key,t in self.tasks.items()}
    def classify(self,role,task,selected,evaluation,features,metadata):
        selected,evaluation=members(selected),members(evaluation);train,test=set(selected),set(evaluation)
        features=tuple(features)
        if not features or len(features)!=len(set(features)) or not set(features)<=set(self.features):raise ValueError('Invalid features')
        if role=='probe':
            spec=self.tasks[task];fold=spec['fold']
        else:
            valid={'main':{f'fold-{i}' for i in range(5)},'replay':{'fold-0'},
                   'audit':{f'main-fold-{i}' for i in range(5)}|{'replay-fold-0'}}
            if role not in valid or task not in valid[role]:raise ValueError('Invalid role/task')
            fold=int(task[-1]);spec=None
        if metadata.get('fold')!=fold:raise ValueError('Operation fold differs from task')
        if metadata.get('role',role)!=role or metadata.get('task',task)!=task:raise ValueError('Operation role/task mismatch')
        outer=self.outers[fold];pool=self.outer_sets[fold]['pool'];original=self.outer_sets[fold]['original_train']
        phase=metadata.get('phase');refs={'main':'reference','replay':'reference','audit':'audit_reference','probe':'reference_timing'}
        if phase==refs[role]:
            if train!=original or features!=self.features:raise ValueError('Reference requires full original train and all features')
            expected=self.task_predictions[task] if spec else self.outer_sets[fold]['test']
            if test!=expected or (spec and spec['kind']!='full_reference_timing'):raise ValueError('Reference evaluation/task differs')
            bucket='reference'
        else:
            allowed={'main':{'utility','candidate','final'},'replay':{'utility','candidate','final'},
                     'audit':{'audit_utility','audit_final'},'probe':{'utility','candidate'}}
            if phase not in allowed[role] or not train<=pool or train & test:raise ValueError('Pool role/phase/membership invalid')
            if spec and spec['kind']!='selector_timing':raise ValueError('Pool operation in reference task')
            if phase in ('utility','audit_utility'):
                context_id=spec['partition_id'] if spec else metadata.get('context')
                context=self.parts['contexts'][context_id]
                if context['fold']!=fold:raise ValueError('Utility context crosses fold')
                resample=spec['resample_index'] if spec else metadata.get('resample')
                method=spec['method'] if spec else metadata.get('method')
                raw,valid=self.utility_sets[context_id,resample,metadata.get('utility_fold')]
                mask=sum(1<<self.features.index(f) for f in features)
                allowed_masks=spec['masks'] if spec else self.mask_sets[context_id,resample,method]
                if method not in ('full','robust_half') or mask not in allowed_masks or metadata.get('mask')!=mask:raise ValueError('Undeclared utility coalition')
                if not train<=raw or test!=valid or (method=='full' and train!=raw):raise ValueError('Utility membership mismatch')
            elif phase=='candidate':
                context=self.parts['contexts'][spec['partition_id'] if spec else outer['inner_context_ids'][metadata['inner']]]
                method=spec['method'] if spec else metadata.get('method')
                raw=self.context_sets[context['partition_id']]['train'];valid=self.task_predictions[task] if spec else self.context_sets[context['partition_id']]['validation']
                if method not in ('full','robust_half') or not train<=raw or test!=valid or (method=='full' and train!=raw):raise ValueError('Candidate membership mismatch')
            elif test!=self.outer_sets[fold]['test']:raise ValueError('Final must use complete outer test')
            bucket='pool'
        if metadata.get('classifier_bucket',bucket)!=bucket or 'binding_token' in metadata:raise ValueError('Caller supplied resource identity')
        return {**metadata,'role':role,'task':task,'classifier_bucket':bucket,
                'training_members_sha256':digest(selected),'evaluation_members_sha256':digest(evaluation),
                'features_sha256':digest(features),'training_rows':len(selected),'evaluation_rows':len(evaluation)}
