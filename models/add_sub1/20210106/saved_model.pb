'
Õ	§	
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.14.02v1.14.0-rc1-22-gaf24dc91b5
X
input_xPlaceholder*
shape
:*
dtype0*
_output_shapes

:
X
input_yPlaceholder*
_output_shapes

:*
dtype0*
shape
:
l
var_z/initial_valueConst*
dtype0*!
valueB"  ?  ?*
_output_shapes

:
y
var_z
VariableV2*
	container *
_output_shapes

:*
dtype0*
shared_name *
shape
:

var_z/AssignAssignvar_zvar_z/initial_value*
_output_shapes

:*
_class

loc:@var_z*
use_locking(*
validate_shape(*
T0
`

var_z/readIdentityvar_z*
_output_shapes

:*
_class

loc:@var_z*
T0
E
addAddinput_xinput_y*
_output_shapes

:*
T0
B
output_aIdentityadd*
T0*
_output_shapes

:
E
subSubinput_xinput_y*
T0*
_output_shapes

:
B
output_bIdentitysub*
T0*
_output_shapes

:

initNoOp^var_z/Assign
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0

save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_9e94c5e1f94c402ca82fae7a023bf4f4/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
\
save/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
f
save/SaveV2/tensor_namesConst*
valueBBvar_z*
dtype0*
_output_shapes
:
e
save/SaveV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
{
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesvar_z*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
N*
T0*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
i
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBvar_z
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/AssignAssignvar_zsave/RestoreV2*
validate_shape(*
T0*
_class

loc:@var_z*
_output_shapes

:*
use_locking(
(
save/restore_shardNoOp^save/Assign
-
save/restore_allNoOp^save/restore_shard "&<
save/Const:0save/Identity:0save/restore_all (5 @F8"Y
trainable_variablesB@
>
var_z:0var_z/Assignvar_z/read:02var_z/initial_value:08"O
	variablesB@
>
var_z:0var_z/Assignvar_z/read:02var_z/initial_value:08*¾
serving_defaultª
"
input_x
	input_x:0
"
input_y
	input_y:0!

result_sub
sub:0!

result_add
add:0tensorflow/serving/predict