(
Њ§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ЄЦ&

rnn_densef/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*"
shared_namernn_densef/kernel
x
%rnn_densef/kernel/Read/ReadVariableOpReadVariableOprnn_densef/kernel*
_output_shapes
:	Р*
dtype0
v
rnn_densef/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namernn_densef/bias
o
#rnn_densef/bias/Read/ReadVariableOpReadVariableOprnn_densef/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_namelstm_lstm/lstm_cell/kernel

.lstm_lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_lstm/lstm_cell/kernel*
_output_shapes

:@*
dtype0
Є
$lstm_lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*5
shared_name&$lstm_lstm/lstm_cell/recurrent_kernel

8lstm_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp$lstm_lstm/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0

lstm_lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelstm_lstm/lstm_cell/bias

,lstm_lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_lstm/lstm_cell/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/rnn_densef/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*)
shared_nameAdam/rnn_densef/kernel/m

,Adam/rnn_densef/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn_densef/kernel/m*
_output_shapes
:	Р*
dtype0

Adam/rnn_densef/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/rnn_densef/bias/m
}
*Adam/rnn_densef/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn_densef/bias/m*
_output_shapes
:*
dtype0

!Adam/lstm_lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!Adam/lstm_lstm/lstm_cell/kernel/m

5Adam/lstm_lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/lstm_lstm/lstm_cell/kernel/m*
_output_shapes

:@*
dtype0
В
+Adam/lstm_lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/lstm_lstm/lstm_cell/recurrent_kernel/m
Ћ
?Adam/lstm_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/lstm_lstm/lstm_cell/recurrent_kernel/m*
_output_shapes

:@*
dtype0

Adam/lstm_lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/lstm_lstm/lstm_cell/bias/m

3Adam/lstm_lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_lstm/lstm_cell/bias/m*
_output_shapes
:@*
dtype0

Adam/rnn_densef/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*)
shared_nameAdam/rnn_densef/kernel/v

,Adam/rnn_densef/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn_densef/kernel/v*
_output_shapes
:	Р*
dtype0

Adam/rnn_densef/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/rnn_densef/bias/v
}
*Adam/rnn_densef/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn_densef/bias/v*
_output_shapes
:*
dtype0

!Adam/lstm_lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!Adam/lstm_lstm/lstm_cell/kernel/v

5Adam/lstm_lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/lstm_lstm/lstm_cell/kernel/v*
_output_shapes

:@*
dtype0
В
+Adam/lstm_lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/lstm_lstm/lstm_cell/recurrent_kernel/v
Ћ
?Adam/lstm_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/lstm_lstm/lstm_cell/recurrent_kernel/v*
_output_shapes

:@*
dtype0

Adam/lstm_lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/lstm_lstm/lstm_cell/bias/v

3Adam/lstm_lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_lstm/lstm_cell/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ь$
valueТ$BП$ BИ$
й
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
 
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api

iter

beta_1

beta_2
	decay
learning_ratemLmM mN!mO"mPvQvR vS!vT"vU
#
 0
!1
"2
3
4
#
 0
!1
"2
3
4
 
­
	variables
#layer_metrics
trainable_variables
$non_trainable_variables
%layer_regularization_losses
regularization_losses

&layers
'metrics
 
~

 kernel
!recurrent_kernel
"bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
 

 0
!1
"2

 0
!1
"2
 
Й
	variables
,layer_metrics
trainable_variables

-states
.non_trainable_variables
/layer_regularization_losses
regularization_losses

0layers
1metrics
 
 
 
­
2layer_metrics
trainable_variables
regularization_losses
3non_trainable_variables
4layer_regularization_losses
	variables

5layers
6metrics
][
VARIABLE_VALUErnn_densef/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErnn_densef/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
7layer_metrics
trainable_variables
regularization_losses
8non_trainable_variables
9layer_regularization_losses
	variables

:layers
;metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElstm_lstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$lstm_lstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUElstm_lstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
3

<0
=1

 0
!1
"2
 

 0
!1
"2
­
>layer_metrics
(trainable_variables
)regularization_losses
?non_trainable_variables
@layer_regularization_losses
*	variables

Alayers
Bmetrics
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
4
	Ctotal
	Dcount
E	variables
F	keras_api
D
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

E	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

J	variables
~
VARIABLE_VALUEAdam/rnn_densef/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/rnn_densef/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_lstm/lstm_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+Adam/lstm_lstm/lstm_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_lstm/lstm_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/rnn_densef/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/rnn_densef/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_lstm/lstm_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+Adam/lstm_lstm/lstm_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_lstm/lstm_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lstm_lstm/lstm_cell/kernellstm_lstm/lstm_cell/bias$lstm_lstm/lstm_cell/recurrent_kernelrnn_densef/kernelrnn_densef/bias*
Tin

2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_506728
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 


StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%rnn_densef/kernel/Read/ReadVariableOp#rnn_densef/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp.lstm_lstm/lstm_cell/kernel/Read/ReadVariableOp8lstm_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp,lstm_lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/rnn_densef/kernel/m/Read/ReadVariableOp*Adam/rnn_densef/bias/m/Read/ReadVariableOp5Adam/lstm_lstm/lstm_cell/kernel/m/Read/ReadVariableOp?Adam/lstm_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp3Adam/lstm_lstm/lstm_cell/bias/m/Read/ReadVariableOp,Adam/rnn_densef/kernel/v/Read/ReadVariableOp*Adam/rnn_densef/bias/v/Read/ReadVariableOp5Adam/lstm_lstm/lstm_cell/kernel/v/Read/ReadVariableOp?Adam/lstm_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp3Adam/lstm_lstm/lstm_cell/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_save_509316
З
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamernn_densef/kernelrnn_densef/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_lstm/lstm_cell/kernel$lstm_lstm/lstm_cell/recurrent_kernellstm_lstm/lstm_cell/biastotalcounttotal_1count_1Adam/rnn_densef/kernel/mAdam/rnn_densef/bias/m!Adam/lstm_lstm/lstm_cell/kernel/m+Adam/lstm_lstm/lstm_cell/recurrent_kernel/mAdam/lstm_lstm/lstm_cell/bias/mAdam/rnn_densef/kernel/vAdam/rnn_densef/bias/v!Adam/lstm_lstm/lstm_cell/kernel/v+Adam/lstm_lstm/lstm_cell/recurrent_kernel/vAdam/lstm_lstm/lstm_cell/bias/v*$
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__traced_restore_509400щк%
и+

A__inference_model_layer_call_and_return_conditional_losses_506590
input_1
lstm_lstm_506560
lstm_lstm_506562
lstm_lstm_506564
rnn_densef_506568
rnn_densef_506570
identityЂ!lstm_lstm/StatefulPartitionedCallЂ"rnn_densef/StatefulPartitionedCall
!lstm_lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_lstm_506560lstm_lstm_506562lstm_lstm_506564*
Tin
2*
Tout
2*+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_5064692#
!lstm_lstm/StatefulPartitionedCallд
flatten/PartitionedCallPartitionedCall*lstm_lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5065052
flatten/PartitionedCall
"rnn_densef/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0rnn_densef_506568rnn_densef_506570*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_rnn_densef_layer_call_and_return_conditional_losses_5065242$
"rnn_densef/StatefulPartitionedCallЧ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_506560*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addл
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_506564*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addШ
IdentityIdentity+rnn_densef/StatefulPartitionedCall:output:0"^lstm_lstm/StatefulPartitionedCall#^rnn_densef/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::2F
!lstm_lstm/StatefulPartitionedCall!lstm_lstm/StatefulPartitionedCall2H
"rnn_densef/StatefulPartitionedCall"rnn_densef/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 
ђ
while_body_505554
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_505578_0
lstm_cell_505580_0
lstm_cell_505582_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_505578
lstm_cell_505580
lstm_cell_505582Ђ!lstm_cell/StatefulPartitionedCallЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem§
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_505578_0lstm_cell_505580_0lstm_cell_505582_0*
Tin

2*
Tout
2*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_5051282#
!lstm_cell/StatefulPartitionedCallж
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3І

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4І

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"&
lstm_cell_505578lstm_cell_505578_0"&
lstm_cell_505580lstm_cell_505580_0"&
lstm_cell_505582lstm_cell_505582_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ы
я
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_506198

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout/ConstЇ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeё
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Охк20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 lstm_cell/dropout/GreaterEqual/yц
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/CastЂ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeї
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2йЕМ22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_1/GreaterEqual/yю
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_1/GreaterEqualЃ
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/CastЊ
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeї
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЋЌ22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_2/GreaterEqual/yю
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_2/GreaterEqualЃ
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/CastЊ
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeї
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Ё22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_3/GreaterEqual/yю
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_3/GreaterEqualЃ
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/CastЊ
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul_1x
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_4/ConstЏ
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeї
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЩЁЂ22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_4/GreaterEqual/yю
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_4/GreaterEqualЃ
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/CastЊ
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_5/ConstЏ
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeі
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2НЯM22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_5/GreaterEqual/yю
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_5/GreaterEqualЃ
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/CastЊ
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_6/ConstЏ
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeї
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2уц22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_6/GreaterEqual/yю
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_6/GreaterEqualЃ
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/CastЊ
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_7/ConstЏ
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeі
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ђ?22
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_7/GreaterEqual/yю
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_7/GreaterEqualЃ
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/CastЊ
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЈ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЊ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterл
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_505982*
condR
while_cond_505981*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeо
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addь
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ё

*__inference_lstm_lstm_layer_call_fn_508846
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_5057872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
мd

E__inference_lstm_cell_layer_call_and_return_conditional_losses_505228

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_3e
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_4e
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_5e
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_6e
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_7x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_8|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Reluh
mul_9MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relu_1n
mul_10MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_10д
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addт
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add^
IdentityIdentity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityb

Identity_1Identity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ј
D
(__inference_flatten_layer_call_fn_508857

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5065052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ыЧ

lstm_lstm_while_body_506911 
lstm_lstm_while_loop_counter&
"lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_lstm_strided_slice_1_0[
Wtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_lstm_strided_slice_1Y
Utensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resourceЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeП
#TensorArrayV2Read/TensorListGetItemTensorListGetItemWtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
lstm_cell/ones_like/ShapeShape*TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout/ConstЇ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape№
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2иёk20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 lstm_cell/dropout/GreaterEqual/yц
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/CastЂ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeї
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЛБ22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_1/GreaterEqual/yю
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_1/GreaterEqualЃ
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/CastЊ
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeі
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЈИN22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_2/GreaterEqual/yю
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_2/GreaterEqualЃ
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/CastЊ
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeї
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ѓ22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_3/GreaterEqual/yю
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_3/GreaterEqualЃ
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/CastЊ
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul_1w
lstm_cell/ones_like_1/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_4/ConstЏ
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeї
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЌоЛ22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_4/GreaterEqual/yю
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_4/GreaterEqualЃ
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/CastЊ
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_5/ConstЏ
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeі
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2РW22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_5/GreaterEqual/yю
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_5/GreaterEqualЃ
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/CastЊ
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_6/ConstЏ
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeі
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2иў22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_6/GreaterEqual/yю
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_6/GreaterEqualЃ
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/CastЊ
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_7/ConstЏ
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeї
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed222
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_7/GreaterEqual/yю
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_7/GreaterEqualЃ
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/CastЊ
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul_1 
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mulІ
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1І
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2І
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЊ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЌ
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10Р
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yh
add_1AddV2lstm_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityi

Identity_1Identity"lstm_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3l

Identity_4Identitylstm_cell/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"8
lstm_lstm_strided_slice_1lstm_lstm_strided_slice_1_0"А
Utensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensorWtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
§

+__inference_rnn_densef_layer_call_fn_508877

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_rnn_densef_layer_call_and_return_conditional_losses_5065242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Г
_
C__inference_flatten_layer_call_and_return_conditional_losses_506505

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ж
Љ
&__inference_model_layer_call_fn_506687
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5066742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
э
Ъ
*__inference_lstm_cell_layer_call_fn_509191

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_5052282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ћ
while_cond_508671
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_508671___redundant_placeholder0.
*while_cond_508671___redundant_placeholder1.
*while_cond_508671___redundant_placeholder2.
*while_cond_508671___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
цЌ

E__inference_lstm_cell_layer_call_and_return_conditional_losses_509057

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeв
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Оќ.2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeй
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2пУ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
dropout_1/GreaterEqual/yЦ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeй
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ГФц2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
dropout_2/GreaterEqual/yЦ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeй
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ќЇЫ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
dropout_3/GreaterEqual/yЦ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_3/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shapeй
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2прЦ2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
dropout_4/GreaterEqual/yЦ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shapeи
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Џ52(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
dropout_5/GreaterEqual/yЦ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
dropout_6/Const
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shapeй
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2нй2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
dropout_6/GreaterEqual/yЦ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_6/GreaterEqual
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_6/Cast
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
dropout_7/Const
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shapeи
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ьЕ;2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
dropout_7/GreaterEqual/yЦ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_7/GreaterEqual
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_7/Cast
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_3f
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_4f
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_5f
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_6f
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_7x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_8|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Reluh
mul_9MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relu_1n
mul_10MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_10д
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addт
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add^
IdentityIdentity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityb

Identity_1Identity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Џ
я
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_506469

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likex
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЈ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЊ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterл
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_506317*
condR
while_cond_506316*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeо
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addь
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
љ_
в
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_505787

inputs
lstm_cell_505689
lstm_cell_505691
lstm_cell_505693
identityЂ!lstm_cell/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2щ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_505689lstm_cell_505691lstm_cell_505693*
Tin

2*
Tout
2*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_5052282#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_505689lstm_cell_505691lstm_cell_505693*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_505702*
condR
while_cond_505701*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeЧ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_cell_505689*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addл
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_cell_505693*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
p

lstm_lstm_while_body_507255 
lstm_lstm_while_loop_counter&
"lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_lstm_strided_slice_1_0[
Wtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_lstm_strided_slice_1Y
Utensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resourceЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeП
#TensorArrayV2Read/TensorListGetItemTensorListGetItemWtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
lstm_cell/ones_like/ShapeShape*TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/ones_like_1/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1Ё
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mulЅ
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1Ѕ
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2Ѕ
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЊ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЌ
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10Р
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yh
add_1AddV2lstm_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityi

Identity_1Identity"lstm_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3l

Identity_4Identitylstm_cell/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"8
lstm_lstm_strided_slice_1lstm_lstm_strided_slice_1_0"А
Utensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensorWtensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
э
Ъ
*__inference_lstm_cell_layer_call_fn_509174

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_5051282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ћ
while_cond_507979
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_507979___redundant_placeholder0.
*while_cond_507979___redundant_placeholder1.
*while_cond_507979___redundant_placeholder2.
*while_cond_507979___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ч
Ы
lstm_lstm_while_cond_507254 
lstm_lstm_while_loop_counter&
"lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
less_lstm_lstm_strided_slice_18
4lstm_lstm_while_cond_507254___redundant_placeholder08
4lstm_lstm_while_cond_507254___redundant_placeholder18
4lstm_lstm_while_cond_507254___redundant_placeholder28
4lstm_lstm_while_cond_507254___redundant_placeholder3
identity
b
LessLessplaceholderless_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ч
Ы
lstm_lstm_while_cond_506910 
lstm_lstm_while_loop_counter&
"lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3"
less_lstm_lstm_strided_slice_18
4lstm_lstm_while_cond_506910___redundant_placeholder08
4lstm_lstm_while_cond_506910___redundant_placeholder18
4lstm_lstm_while_cond_506910___redundant_placeholder28
4lstm_lstm_while_cond_506910___redundant_placeholder3
identity
b
LessLessplaceholderless_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
е+

A__inference_model_layer_call_and_return_conditional_losses_506674

inputs
lstm_lstm_506644
lstm_lstm_506646
lstm_lstm_506648
rnn_densef_506652
rnn_densef_506654
identityЂ!lstm_lstm/StatefulPartitionedCallЂ"rnn_densef/StatefulPartitionedCall
!lstm_lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_506644lstm_lstm_506646lstm_lstm_506648*
Tin
2*
Tout
2*+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_5064692#
!lstm_lstm/StatefulPartitionedCallд
flatten/PartitionedCallPartitionedCall*lstm_lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5065052
flatten/PartitionedCall
"rnn_densef/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0rnn_densef_506652rnn_densef_506654*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_rnn_densef_layer_call_and_return_conditional_losses_5065242$
"rnn_densef/StatefulPartitionedCallЧ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_506644*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addл
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_506648*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addШ
IdentityIdentity+rnn_densef/StatefulPartitionedCall:output:0"^lstm_lstm/StatefulPartitionedCall#^rnn_densef/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::2F
!lstm_lstm/StatefulPartitionedCall!lstm_lstm/StatefulPartitionedCall2H
"rnn_densef/StatefulPartitionedCall"rnn_densef/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ж
Љ
&__inference_model_layer_call_fn_506639
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5066262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
ћ
!model_lstm_lstm_while_cond_504778&
"model_lstm_lstm_while_loop_counter,
(model_lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3(
$less_model_lstm_lstm_strided_slice_1>
:model_lstm_lstm_while_cond_504778___redundant_placeholder0>
:model_lstm_lstm_while_cond_504778___redundant_placeholder1>
:model_lstm_lstm_while_cond_504778___redundant_placeholder2>
:model_lstm_lstm_while_cond_504778___redundant_placeholder3
identity
h
LessLessplaceholder$less_model_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ч

*__inference_lstm_lstm_layer_call_fn_508143

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_5061982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г
Ј
&__inference_model_layer_call_fn_507446

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5066742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
зЌ

E__inference_lstm_cell_layer_call_and_return_conditional_losses_505128

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeв
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2аr2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeй
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЃЯ§2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
dropout_1/GreaterEqual/yЦ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeй
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Л§2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
dropout_2/GreaterEqual/yЦ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeй
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2л2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
dropout_3/GreaterEqual/yЦ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_3/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shapeй
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Едю2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
dropout_4/GreaterEqual/yЦ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shapeй
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2г2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
dropout_5/GreaterEqual/yЦ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
dropout_6/Const
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shapeй
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЕОЪ2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
dropout_6/GreaterEqual/yЦ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_6/GreaterEqual
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_6/Cast
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
dropout_7/Const
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shapeи
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2мЫ2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
dropout_7/GreaterEqual/yЦ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_7/GreaterEqual
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_7/Cast
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_3d
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_4d
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_5d
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_6d
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_7x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_8|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Reluh
mul_9MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relu_1n
mul_10MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_10д
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addт
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add^
IdentityIdentity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityb

Identity_1Identity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
 
ђ
while_body_505702
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
lstm_cell_505726_0
lstm_cell_505728_0
lstm_cell_505730_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
lstm_cell_505726
lstm_cell_505728
lstm_cell_505730Ђ!lstm_cell/StatefulPartitionedCallЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem§
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3lstm_cell_505726_0lstm_cell_505728_0lstm_cell_505730_0*
Tin

2*
Tout
2*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_5052282#
!lstm_cell/StatefulPartitionedCallж
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder*lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1p
IdentityIdentity	add_1:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1r

Identity_2Identityadd:z:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0"^lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3І

Identity_4Identity*lstm_cell/StatefulPartitionedCall:output:1"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4І

Identity_5Identity*lstm_cell/StatefulPartitionedCall:output:2"^lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"&
lstm_cell_505726lstm_cell_505726_0"&
lstm_cell_505728lstm_cell_505728_0"&
lstm_cell_505730lstm_cell_505730_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ю@

__inference__traced_save_509316
file_prefix0
,savev2_rnn_densef_kernel_read_readvariableop.
*savev2_rnn_densef_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop9
5savev2_lstm_lstm_lstm_cell_kernel_read_readvariableopC
?savev2_lstm_lstm_lstm_cell_recurrent_kernel_read_readvariableop7
3savev2_lstm_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_rnn_densef_kernel_m_read_readvariableop5
1savev2_adam_rnn_densef_bias_m_read_readvariableop@
<savev2_adam_lstm_lstm_lstm_cell_kernel_m_read_readvariableopJ
Fsavev2_adam_lstm_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop>
:savev2_adam_lstm_lstm_lstm_cell_bias_m_read_readvariableop7
3savev2_adam_rnn_densef_kernel_v_read_readvariableop5
1savev2_adam_rnn_densef_bias_v_read_readvariableop@
<savev2_adam_lstm_lstm_lstm_cell_kernel_v_read_readvariableopJ
Fsavev2_adam_lstm_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop>
:savev2_adam_lstm_lstm_lstm_cell_bias_v_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b6436c726a9348cdb77261d36b30a37e/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesИ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesі

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_rnn_densef_kernel_read_readvariableop*savev2_rnn_densef_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop5savev2_lstm_lstm_lstm_cell_kernel_read_readvariableop?savev2_lstm_lstm_lstm_cell_recurrent_kernel_read_readvariableop3savev2_lstm_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_rnn_densef_kernel_m_read_readvariableop1savev2_adam_rnn_densef_bias_m_read_readvariableop<savev2_adam_lstm_lstm_lstm_cell_kernel_m_read_readvariableopFsavev2_adam_lstm_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop:savev2_adam_lstm_lstm_lstm_cell_bias_m_read_readvariableop3savev2_adam_rnn_densef_kernel_v_read_readvariableop1savev2_adam_rnn_densef_bias_v_read_readvariableop<savev2_adam_lstm_lstm_lstm_cell_kernel_v_read_readvariableopFsavev2_adam_lstm_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop:savev2_adam_lstm_lstm_lstm_cell_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ќ
_input_shapes
: :	Р:: : : : : :@:@:@: : : : :	Р::@:@:@:	Р::@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Р: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@:$	 

_output_shapes

:@: 


_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Р: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@:%!

_output_shapes
:	Р: 

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@:

_output_shapes
: 
ё

*__inference_lstm_lstm_layer_call_fn_508835
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_5056392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
o
а
while_body_508672
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resourceЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
lstm_cell/ones_like/ShapeShape*TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/ones_like_1/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1Ё
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mulЅ
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1Ѕ
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2Ѕ
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЊ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЌ
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10Р
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3l

Identity_4Identitylstm_cell/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
Џ
я
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_508132

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likex
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЈ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЊ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterл
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_507980*
condR
while_cond_507979*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeо
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addь
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Єl
Ц
"__inference__traced_restore_509400
file_prefix&
"assignvariableop_rnn_densef_kernel&
"assignvariableop_1_rnn_densef_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate1
-assignvariableop_7_lstm_lstm_lstm_cell_kernel;
7assignvariableop_8_lstm_lstm_lstm_cell_recurrent_kernel/
+assignvariableop_9_lstm_lstm_lstm_cell_bias
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_10
,assignvariableop_14_adam_rnn_densef_kernel_m.
*assignvariableop_15_adam_rnn_densef_bias_m9
5assignvariableop_16_adam_lstm_lstm_lstm_cell_kernel_mC
?assignvariableop_17_adam_lstm_lstm_lstm_cell_recurrent_kernel_m7
3assignvariableop_18_adam_lstm_lstm_lstm_cell_bias_m0
,assignvariableop_19_adam_rnn_densef_kernel_v.
*assignvariableop_20_adam_rnn_densef_bias_v9
5assignvariableop_21_adam_lstm_lstm_lstm_cell_kernel_vC
?assignvariableop_22_adam_lstm_lstm_lstm_cell_recurrent_kernel_v7
3assignvariableop_23_adam_lstm_lstm_lstm_cell_bias_v
identity_25ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesО
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЃ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp"assignvariableop_rnn_densef_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp"assignvariableop_1_rnn_densef_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOp-assignvariableop_7_lstm_lstm_lstm_cell_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8­
AssignVariableOp_8AssignVariableOp7assignvariableop_8_lstm_lstm_lstm_cell_recurrent_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ё
AssignVariableOp_9AssignVariableOp+assignvariableop_9_lstm_lstm_lstm_cell_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ѕ
AssignVariableOp_14AssignVariableOp,assignvariableop_14_adam_rnn_densef_kernel_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ѓ
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_rnn_densef_bias_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Ў
AssignVariableOp_16AssignVariableOp5assignvariableop_16_adam_lstm_lstm_lstm_cell_kernel_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17И
AssignVariableOp_17AssignVariableOp?assignvariableop_17_adam_lstm_lstm_lstm_cell_recurrent_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ќ
AssignVariableOp_18AssignVariableOp3assignvariableop_18_adam_lstm_lstm_lstm_cell_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ѕ
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_rnn_densef_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_rnn_densef_bias_vIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Ў
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_lstm_lstm_lstm_cell_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22И
AssignVariableOp_22AssignVariableOp?assignvariableop_22_adam_lstm_lstm_lstm_cell_recurrent_kernel_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Ќ
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_lstm_lstm_lstm_cell_bias_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpю
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24ћ
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г
_
C__inference_flatten_layer_call_and_return_conditional_losses_508852

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
фp
Р
!model_lstm_lstm_while_body_504779&
"model_lstm_lstm_while_loop_counter,
(model_lstm_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!model_lstm_lstm_strided_slice_1_0a
]tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5#
model_lstm_lstm_strided_slice_1_
[tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resourceЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeХ
#TensorArrayV2Read/TensorListGetItemTensorListGetItem]tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
lstm_cell/ones_like/ShapeShape*TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/ones_like_1/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1Ё
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mulЅ
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1Ѕ
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2Ѕ
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЊ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЌ
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10Р
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yn
add_1AddV2"model_lstm_lstm_while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identityo

Identity_1Identity(model_lstm_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3l

Identity_4Identitylstm_cell/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"D
model_lstm_lstm_strided_slice_1!model_lstm_lstm_strided_slice_1_0"М
[tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor]tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
ЭЏ
ё
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_508824
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identityЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likex
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЈ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЊ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterл
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_508672*
condR
while_cond_508671*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeо
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addь
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addx
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
и+

A__inference_model_layer_call_and_return_conditional_losses_506557
input_1
lstm_lstm_506492
lstm_lstm_506494
lstm_lstm_506496
rnn_densef_506535
rnn_densef_506537
identityЂ!lstm_lstm/StatefulPartitionedCallЂ"rnn_densef/StatefulPartitionedCall
!lstm_lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_lstm_506492lstm_lstm_506494lstm_lstm_506496*
Tin
2*
Tout
2*+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_5061982#
!lstm_lstm/StatefulPartitionedCallд
flatten/PartitionedCallPartitionedCall*lstm_lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5065052
flatten/PartitionedCall
"rnn_densef/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0rnn_densef_506535rnn_densef_506537*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_rnn_densef_layer_call_and_return_conditional_losses_5065242$
"rnn_densef/StatefulPartitionedCallЧ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_506492*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addл
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_506496*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addШ
IdentityIdentity+rnn_densef/StatefulPartitionedCall:output:0"^lstm_lstm/StatefulPartitionedCall#^rnn_densef/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::2F
!lstm_lstm/StatefulPartitionedCall!lstm_lstm/StatefulPartitionedCall2H
"rnn_densef/StatefulPartitionedCall"rnn_densef/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ћ
while_cond_507644
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_507644___redundant_placeholder0.
*while_cond_507644___redundant_placeholder1.
*while_cond_507644___redundant_placeholder2.
*while_cond_507644___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
и
ђ
A__inference_model_layer_call_and_return_conditional_losses_507416

inputs5
1lstm_lstm_lstm_cell_split_readvariableop_resource7
3lstm_lstm_lstm_cell_split_1_readvariableop_resource/
+lstm_lstm_lstm_cell_readvariableop_resource-
)rnn_densef_matmul_readvariableop_resource.
*rnn_densef_biasadd_readvariableop_resource
identityЂlstm_lstm/whileX
lstm_lstm/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_lstm/Shape
lstm_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_lstm/strided_slice/stack
lstm_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_lstm/strided_slice/stack_1
lstm_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_lstm/strided_slice/stack_2
lstm_lstm/strided_sliceStridedSlicelstm_lstm/Shape:output:0&lstm_lstm/strided_slice/stack:output:0(lstm_lstm/strided_slice/stack_1:output:0(lstm_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_lstm/strided_slicep
lstm_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros/mul/y
lstm_lstm/zeros/mulMul lstm_lstm/strided_slice:output:0lstm_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros/muls
lstm_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_lstm/zeros/Less/y
lstm_lstm/zeros/LessLesslstm_lstm/zeros/mul:z:0lstm_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros/Lessv
lstm_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros/packed/1Ћ
lstm_lstm/zeros/packedPack lstm_lstm/strided_slice:output:0!lstm_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_lstm/zeros/packeds
lstm_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/zeros/Const
lstm_lstm/zerosFilllstm_lstm/zeros/packed:output:0lstm_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/zerost
lstm_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros_1/mul/y
lstm_lstm/zeros_1/mulMul lstm_lstm/strided_slice:output:0 lstm_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros_1/mulw
lstm_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_lstm/zeros_1/Less/y
lstm_lstm/zeros_1/LessLesslstm_lstm/zeros_1/mul:z:0!lstm_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros_1/Lessz
lstm_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros_1/packed/1Б
lstm_lstm/zeros_1/packedPack lstm_lstm/strided_slice:output:0#lstm_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_lstm/zeros_1/packedw
lstm_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/zeros_1/ConstЅ
lstm_lstm/zeros_1Fill!lstm_lstm/zeros_1/packed:output:0 lstm_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/zeros_1
lstm_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_lstm/transpose/perm
lstm_lstm/transpose	Transposeinputs!lstm_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_lstm/transposem
lstm_lstm/Shape_1Shapelstm_lstm/transpose:y:0*
T0*
_output_shapes
:2
lstm_lstm/Shape_1
lstm_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_lstm/strided_slice_1/stack
!lstm_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_1/stack_1
!lstm_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_1/stack_2Њ
lstm_lstm/strided_slice_1StridedSlicelstm_lstm/Shape_1:output:0(lstm_lstm/strided_slice_1/stack:output:0*lstm_lstm/strided_slice_1/stack_1:output:0*lstm_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_lstm/strided_slice_1
%lstm_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%lstm_lstm/TensorArrayV2/element_shapeк
lstm_lstm/TensorArrayV2TensorListReserve.lstm_lstm/TensorArrayV2/element_shape:output:0"lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_lstm/TensorArrayV2г
?lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape 
1lstm_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_lstm/transpose:y:0Hlstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1lstm_lstm/TensorArrayUnstack/TensorListFromTensor
lstm_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_lstm/strided_slice_2/stack
!lstm_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_2/stack_1
!lstm_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_2/stack_2И
lstm_lstm/strided_slice_2StridedSlicelstm_lstm/transpose:y:0(lstm_lstm/strided_slice_2/stack:output:0*lstm_lstm/strided_slice_2/stack_1:output:0*lstm_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_lstm/strided_slice_2
#lstm_lstm/lstm_cell/ones_like/ShapeShape"lstm_lstm/strided_slice_2:output:0*
T0*
_output_shapes
:2%
#lstm_lstm/lstm_cell/ones_like/Shape
#lstm_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_lstm/lstm_cell/ones_like/Constд
lstm_lstm/lstm_cell/ones_likeFill,lstm_lstm/lstm_cell/ones_like/Shape:output:0,lstm_lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/ones_like
%lstm_lstm/lstm_cell/ones_like_1/ShapeShapelstm_lstm/zeros:output:0*
T0*
_output_shapes
:2'
%lstm_lstm/lstm_cell/ones_like_1/Shape
%lstm_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%lstm_lstm/lstm_cell/ones_like_1/Constм
lstm_lstm/lstm_cell/ones_like_1Fill.lstm_lstm/lstm_cell/ones_like_1/Shape:output:0.lstm_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
lstm_lstm/lstm_cell/ones_like_1З
lstm_lstm/lstm_cell/mulMul"lstm_lstm/strided_slice_2:output:0&lstm_lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mulЛ
lstm_lstm/lstm_cell/mul_1Mul"lstm_lstm/strided_slice_2:output:0&lstm_lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_1Л
lstm_lstm/lstm_cell/mul_2Mul"lstm_lstm/strided_slice_2:output:0&lstm_lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_2Л
lstm_lstm/lstm_cell/mul_3Mul"lstm_lstm/strided_slice_2:output:0&lstm_lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_3x
lstm_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/lstm_cell/Const
#lstm_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_lstm/lstm_cell/split/split_dimЦ
(lstm_lstm/lstm_cell/split/ReadVariableOpReadVariableOp1lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02*
(lstm_lstm/lstm_cell/split/ReadVariableOpї
lstm_lstm/lstm_cell/splitSplit,lstm_lstm/lstm_cell/split/split_dim:output:00lstm_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_lstm/lstm_cell/splitЕ
lstm_lstm/lstm_cell/MatMulMatMullstm_lstm/lstm_cell/mul:z:0"lstm_lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMulЛ
lstm_lstm/lstm_cell/MatMul_1MatMullstm_lstm/lstm_cell/mul_1:z:0"lstm_lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_1Л
lstm_lstm/lstm_cell/MatMul_2MatMullstm_lstm/lstm_cell/mul_2:z:0"lstm_lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_2Л
lstm_lstm/lstm_cell/MatMul_3MatMullstm_lstm/lstm_cell/mul_3:z:0"lstm_lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_3|
lstm_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/lstm_cell/Const_1
%lstm_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%lstm_lstm/lstm_cell/split_1/split_dimШ
*lstm_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp3lstm_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02,
*lstm_lstm/lstm_cell/split_1/ReadVariableOpя
lstm_lstm/lstm_cell/split_1Split.lstm_lstm/lstm_cell/split_1/split_dim:output:02lstm_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_lstm/lstm_cell/split_1У
lstm_lstm/lstm_cell/BiasAddBiasAdd$lstm_lstm/lstm_cell/MatMul:product:0$lstm_lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/BiasAddЩ
lstm_lstm/lstm_cell/BiasAdd_1BiasAdd&lstm_lstm/lstm_cell/MatMul_1:product:0$lstm_lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/BiasAdd_1Щ
lstm_lstm/lstm_cell/BiasAdd_2BiasAdd&lstm_lstm/lstm_cell/MatMul_2:product:0$lstm_lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/BiasAdd_2Щ
lstm_lstm/lstm_cell/BiasAdd_3BiasAdd&lstm_lstm/lstm_cell/MatMul_3:product:0$lstm_lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/BiasAdd_3Г
lstm_lstm/lstm_cell/mul_4Mullstm_lstm/zeros:output:0(lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_4Г
lstm_lstm/lstm_cell/mul_5Mullstm_lstm/zeros:output:0(lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_5Г
lstm_lstm/lstm_cell/mul_6Mullstm_lstm/zeros:output:0(lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_6Г
lstm_lstm/lstm_cell/mul_7Mullstm_lstm/zeros:output:0(lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_7Д
"lstm_lstm/lstm_cell/ReadVariableOpReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02$
"lstm_lstm/lstm_cell/ReadVariableOpЃ
'lstm_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'lstm_lstm/lstm_cell/strided_slice/stackЇ
)lstm_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)lstm_lstm/lstm_cell/strided_slice/stack_1Ї
)lstm_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)lstm_lstm/lstm_cell/strided_slice/stack_2є
!lstm_lstm/lstm_cell/strided_sliceStridedSlice*lstm_lstm/lstm_cell/ReadVariableOp:value:00lstm_lstm/lstm_cell/strided_slice/stack:output:02lstm_lstm/lstm_cell/strided_slice/stack_1:output:02lstm_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!lstm_lstm/lstm_cell/strided_sliceУ
lstm_lstm/lstm_cell/MatMul_4MatMullstm_lstm/lstm_cell/mul_4:z:0*lstm_lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_4Л
lstm_lstm/lstm_cell/addAddV2$lstm_lstm/lstm_cell/BiasAdd:output:0&lstm_lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add
lstm_lstm/lstm_cell/SigmoidSigmoidlstm_lstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/SigmoidИ
$lstm_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_1Ї
)lstm_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2+
)lstm_lstm/lstm_cell/strided_slice_1/stackЋ
+lstm_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_lstm/lstm_cell/strided_slice_1/stack_1Ћ
+lstm_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_1/stack_2
#lstm_lstm/lstm_cell/strided_slice_1StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_1:value:02lstm_lstm/lstm_cell/strided_slice_1/stack:output:04lstm_lstm/lstm_cell/strided_slice_1/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_1Х
lstm_lstm/lstm_cell/MatMul_5MatMullstm_lstm/lstm_cell/mul_5:z:0,lstm_lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_5С
lstm_lstm/lstm_cell/add_1AddV2&lstm_lstm/lstm_cell/BiasAdd_1:output:0&lstm_lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add_1
lstm_lstm/lstm_cell/Sigmoid_1Sigmoidlstm_lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/Sigmoid_1Ў
lstm_lstm/lstm_cell/mul_8Mul!lstm_lstm/lstm_cell/Sigmoid_1:y:0lstm_lstm/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_8И
$lstm_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_2Ї
)lstm_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_lstm/lstm_cell/strided_slice_2/stackЋ
+lstm_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2-
+lstm_lstm/lstm_cell/strided_slice_2/stack_1Ћ
+lstm_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_2/stack_2
#lstm_lstm/lstm_cell/strided_slice_2StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_2:value:02lstm_lstm/lstm_cell/strided_slice_2/stack:output:04lstm_lstm/lstm_cell/strided_slice_2/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_2Х
lstm_lstm/lstm_cell/MatMul_6MatMullstm_lstm/lstm_cell/mul_6:z:0,lstm_lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_6С
lstm_lstm/lstm_cell/add_2AddV2&lstm_lstm/lstm_cell/BiasAdd_2:output:0&lstm_lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add_2
lstm_lstm/lstm_cell/ReluRelulstm_lstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/ReluИ
lstm_lstm/lstm_cell/mul_9Mullstm_lstm/lstm_cell/Sigmoid:y:0&lstm_lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_9Џ
lstm_lstm/lstm_cell/add_3AddV2lstm_lstm/lstm_cell/mul_8:z:0lstm_lstm/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add_3И
$lstm_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_3Ї
)lstm_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2+
)lstm_lstm/lstm_cell/strided_slice_3/stackЋ
+lstm_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_lstm/lstm_cell/strided_slice_3/stack_1Ћ
+lstm_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_3/stack_2
#lstm_lstm/lstm_cell/strided_slice_3StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_3:value:02lstm_lstm/lstm_cell/strided_slice_3/stack:output:04lstm_lstm/lstm_cell/strided_slice_3/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_3Х
lstm_lstm/lstm_cell/MatMul_7MatMullstm_lstm/lstm_cell/mul_7:z:0,lstm_lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_7С
lstm_lstm/lstm_cell/add_4AddV2&lstm_lstm/lstm_cell/BiasAdd_3:output:0&lstm_lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add_4
lstm_lstm/lstm_cell/Sigmoid_2Sigmoidlstm_lstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/Sigmoid_2
lstm_lstm/lstm_cell/Relu_1Relulstm_lstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/Relu_1О
lstm_lstm/lstm_cell/mul_10Mul!lstm_lstm/lstm_cell/Sigmoid_2:y:0(lstm_lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_10Ѓ
'lstm_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2)
'lstm_lstm/TensorArrayV2_1/element_shapeр
lstm_lstm/TensorArrayV2_1TensorListReserve0lstm_lstm/TensorArrayV2_1/element_shape:output:0"lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_lstm/TensorArrayV2_1b
lstm_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_lstm/time
"lstm_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"lstm_lstm/while/maximum_iterations~
lstm_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_lstm/while/loop_counterё
lstm_lstm/whileWhile%lstm_lstm/while/loop_counter:output:0+lstm_lstm/while/maximum_iterations:output:0lstm_lstm/time:output:0"lstm_lstm/TensorArrayV2_1:handle:0lstm_lstm/zeros:output:0lstm_lstm/zeros_1:output:0"lstm_lstm/strided_slice_1:output:0Alstm_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_lstm_lstm_cell_split_readvariableop_resource3lstm_lstm_lstm_cell_split_1_readvariableop_resource+lstm_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_lstm_while_body_507255*'
condR
lstm_lstm_while_cond_507254*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
lstm_lstm/whileЩ
:lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2<
:lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape
,lstm_lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm_lstm/while:output:3Clstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02.
,lstm_lstm/TensorArrayV2Stack/TensorListStack
lstm_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2!
lstm_lstm/strided_slice_3/stack
!lstm_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!lstm_lstm/strided_slice_3/stack_1
!lstm_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_3/stack_2ж
lstm_lstm/strided_slice_3StridedSlice5lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0(lstm_lstm/strided_slice_3/stack:output:0*lstm_lstm/strided_slice_3/stack_1:output:0*lstm_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_lstm/strided_slice_3
lstm_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_lstm/transpose_1/permЭ
lstm_lstm/transpose_1	Transpose5lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_lstm/transpose_1z
lstm_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/runtimeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten/Const
flatten/ReshapeReshapelstm_lstm/transpose_1:y:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/ReshapeЏ
 rnn_densef/MatMul/ReadVariableOpReadVariableOp)rnn_densef_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02"
 rnn_densef/MatMul/ReadVariableOpІ
rnn_densef/MatMulMatMulflatten/Reshape:output:0(rnn_densef/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
rnn_densef/MatMul­
!rnn_densef/BiasAdd/ReadVariableOpReadVariableOp*rnn_densef_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rnn_densef/BiasAdd/ReadVariableOp­
rnn_densef/BiasAddBiasAddrnn_densef/MatMul:product:0)rnn_densef/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
rnn_densef/BiasAdd
rnn_densef/SoftmaxSoftmaxrnn_densef/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
rnn_densef/Softmaxш
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp1lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addі
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add
IdentityIdentityrnn_densef/Softmax:softmax:0^lstm_lstm/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::2"
lstm_lstm/whilelstm_lstm/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Щ
w
__inference_loss_fn_0_509204F
Blstm_lstm_lstm_cell_kernel_regularizer_abs_readvariableop_resource
identityљ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBlstm_lstm_lstm_cell_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addq
IdentityIdentity.lstm_lstm/lstm_cell/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ђ
Ў
F__inference_rnn_densef_layer_call_and_return_conditional_losses_506524

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
уЦ
а
while_body_508337
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resourceЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
lstm_cell/ones_like/ShapeShape*TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout/ConstЇ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeё
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ўЕј20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 lstm_cell/dropout/GreaterEqual/yц
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/CastЂ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeї
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2уЖ22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_1/GreaterEqual/yю
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_1/GreaterEqualЃ
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/CastЊ
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeї
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2цЦЅ22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_2/GreaterEqual/yю
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_2/GreaterEqualЃ
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/CastЊ
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeї
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Жч22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_3/GreaterEqual/yю
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_3/GreaterEqualЃ
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/CastЊ
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul_1w
lstm_cell/ones_like_1/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_4/ConstЏ
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeї
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЈЭ22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_4/GreaterEqual/yю
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_4/GreaterEqualЃ
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/CastЊ
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_5/ConstЏ
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeї
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2йЎ22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_5/GreaterEqual/yю
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_5/GreaterEqualЃ
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/CastЊ
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_6/ConstЏ
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeї
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЉМ22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_6/GreaterEqual/yю
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_6/GreaterEqualЃ
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/CastЊ
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_7/ConstЏ
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeї
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ћЖс22
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_7/GreaterEqual/yю
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_7/GreaterEqualЃ
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/CastЊ
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul_1 
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mulІ
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1І
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2І
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЊ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЌ
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10Р
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3l

Identity_4Identitylstm_cell/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 

ћ
while_cond_505553
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_505553___redundant_placeholder0.
*while_cond_505553___redundant_placeholder1.
*while_cond_505553___redundant_placeholder2.
*while_cond_505553___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ћ
ё
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_508553
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identityЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout/ConstЇ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeё
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Ћ§20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 lstm_cell/dropout/GreaterEqual/yц
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/CastЂ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeї
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2чюл22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_1/GreaterEqual/yю
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_1/GreaterEqualЃ
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/CastЊ
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeї
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ееа22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_2/GreaterEqual/yю
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_2/GreaterEqualЃ
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/CastЊ
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeї
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2кП22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_3/GreaterEqual/yю
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_3/GreaterEqualЃ
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/CastЊ
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul_1x
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_4/ConstЏ
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeі
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2т§22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_4/GreaterEqual/yю
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_4/GreaterEqualЃ
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/CastЊ
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_5/ConstЏ
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeї
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Эым22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_5/GreaterEqual/yю
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_5/GreaterEqualЃ
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/CastЊ
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_6/ConstЏ
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeї
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2еЩ22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_6/GreaterEqual/yю
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_6/GreaterEqualЃ
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/CastЊ
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_7/ConstЏ
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeї
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2іІ22
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_7/GreaterEqual/yю
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_7/GreaterEqualЃ
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/CastЊ
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЈ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЊ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterл
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_508337*
condR
while_cond_508336*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeо
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addь
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addx
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ћ
while_cond_508336
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_508336___redundant_placeholder0.
*while_cond_508336___redundant_placeholder1.
*while_cond_508336___redundant_placeholder2.
*while_cond_508336___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
o
а
while_body_507980
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resourceЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
lstm_cell/ones_like/ShapeShape*TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/ones_like_1/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1Ё
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mulЅ
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1Ѕ
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2Ѕ
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЊ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЌ
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10Р
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3l

Identity_4Identitylstm_cell/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
љ_
в
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_505639

inputs
lstm_cell_505541
lstm_cell_505543
lstm_cell_505545
identityЂ!lstm_cell/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2щ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_505541lstm_cell_505543lstm_cell_505545*
Tin

2*
Tout
2*M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_5051282#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_505541lstm_cell_505543lstm_cell_505545*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_505554*
condR
while_cond_505553*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeЧ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_cell_505541*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addл
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_cell_505545*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
џЯ
ї
!__inference__wrapped_model_504924
input_1;
7model_lstm_lstm_lstm_cell_split_readvariableop_resource=
9model_lstm_lstm_lstm_cell_split_1_readvariableop_resource5
1model_lstm_lstm_lstm_cell_readvariableop_resource3
/model_rnn_densef_matmul_readvariableop_resource4
0model_rnn_densef_biasadd_readvariableop_resource
identityЂmodel/lstm_lstm/whilee
model/lstm_lstm/ShapeShapeinput_1*
T0*
_output_shapes
:2
model/lstm_lstm/Shape
#model/lstm_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model/lstm_lstm/strided_slice/stack
%model/lstm_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/lstm_lstm/strided_slice/stack_1
%model/lstm_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/lstm_lstm/strided_slice/stack_2Т
model/lstm_lstm/strided_sliceStridedSlicemodel/lstm_lstm/Shape:output:0,model/lstm_lstm/strided_slice/stack:output:0.model/lstm_lstm/strided_slice/stack_1:output:0.model/lstm_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/lstm_lstm/strided_slice|
model/lstm_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/lstm_lstm/zeros/mul/yЌ
model/lstm_lstm/zeros/mulMul&model/lstm_lstm/strided_slice:output:0$model/lstm_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm_lstm/zeros/mul
model/lstm_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
model/lstm_lstm/zeros/Less/yЇ
model/lstm_lstm/zeros/LessLessmodel/lstm_lstm/zeros/mul:z:0%model/lstm_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm_lstm/zeros/Less
model/lstm_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
model/lstm_lstm/zeros/packed/1У
model/lstm_lstm/zeros/packedPack&model/lstm_lstm/strided_slice:output:0'model/lstm_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/lstm_lstm/zeros/packed
model/lstm_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm_lstm/zeros/ConstЕ
model/lstm_lstm/zerosFill%model/lstm_lstm/zeros/packed:output:0$model/lstm_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/lstm_lstm/zeros
model/lstm_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/lstm_lstm/zeros_1/mul/yВ
model/lstm_lstm/zeros_1/mulMul&model/lstm_lstm/strided_slice:output:0&model/lstm_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm_lstm/zeros_1/mul
model/lstm_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
model/lstm_lstm/zeros_1/Less/yЏ
model/lstm_lstm/zeros_1/LessLessmodel/lstm_lstm/zeros_1/mul:z:0'model/lstm_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm_lstm/zeros_1/Less
 model/lstm_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/lstm_lstm/zeros_1/packed/1Щ
model/lstm_lstm/zeros_1/packedPack&model/lstm_lstm/strided_slice:output:0)model/lstm_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model/lstm_lstm/zeros_1/packed
model/lstm_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm_lstm/zeros_1/ConstН
model/lstm_lstm/zeros_1Fill'model/lstm_lstm/zeros_1/packed:output:0&model/lstm_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/lstm_lstm/zeros_1
model/lstm_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model/lstm_lstm/transpose/permЋ
model/lstm_lstm/transpose	Transposeinput_1'model/lstm_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
model/lstm_lstm/transpose
model/lstm_lstm/Shape_1Shapemodel/lstm_lstm/transpose:y:0*
T0*
_output_shapes
:2
model/lstm_lstm/Shape_1
%model/lstm_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model/lstm_lstm/strided_slice_1/stack
'model/lstm_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/lstm_lstm/strided_slice_1/stack_1
'model/lstm_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/lstm_lstm/strided_slice_1/stack_2Ю
model/lstm_lstm/strided_slice_1StridedSlice model/lstm_lstm/Shape_1:output:0.model/lstm_lstm/strided_slice_1/stack:output:00model/lstm_lstm/strided_slice_1/stack_1:output:00model/lstm_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model/lstm_lstm/strided_slice_1Ѕ
+model/lstm_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+model/lstm_lstm/TensorArrayV2/element_shapeђ
model/lstm_lstm/TensorArrayV2TensorListReserve4model/lstm_lstm/TensorArrayV2/element_shape:output:0(model/lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/lstm_lstm/TensorArrayV2п
Emodel/lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Emodel/lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7model/lstm_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/lstm_lstm/transpose:y:0Nmodel/lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7model/lstm_lstm/TensorArrayUnstack/TensorListFromTensor
%model/lstm_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model/lstm_lstm/strided_slice_2/stack
'model/lstm_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/lstm_lstm/strided_slice_2/stack_1
'model/lstm_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/lstm_lstm/strided_slice_2/stack_2м
model/lstm_lstm/strided_slice_2StridedSlicemodel/lstm_lstm/transpose:y:0.model/lstm_lstm/strided_slice_2/stack:output:00model/lstm_lstm/strided_slice_2/stack_1:output:00model/lstm_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
model/lstm_lstm/strided_slice_2Ў
)model/lstm_lstm/lstm_cell/ones_like/ShapeShape(model/lstm_lstm/strided_slice_2:output:0*
T0*
_output_shapes
:2+
)model/lstm_lstm/lstm_cell/ones_like/Shape
)model/lstm_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)model/lstm_lstm/lstm_cell/ones_like/Constь
#model/lstm_lstm/lstm_cell/ones_likeFill2model/lstm_lstm/lstm_cell/ones_like/Shape:output:02model/lstm_lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#model/lstm_lstm/lstm_cell/ones_likeЈ
+model/lstm_lstm/lstm_cell/ones_like_1/ShapeShapemodel/lstm_lstm/zeros:output:0*
T0*
_output_shapes
:2-
+model/lstm_lstm/lstm_cell/ones_like_1/Shape
+model/lstm_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+model/lstm_lstm/lstm_cell/ones_like_1/Constє
%model/lstm_lstm/lstm_cell/ones_like_1Fill4model/lstm_lstm/lstm_cell/ones_like_1/Shape:output:04model/lstm_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2'
%model/lstm_lstm/lstm_cell/ones_like_1Я
model/lstm_lstm/lstm_cell/mulMul(model/lstm_lstm/strided_slice_2:output:0,model/lstm_lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/lstm_lstm/lstm_cell/mulг
model/lstm_lstm/lstm_cell/mul_1Mul(model/lstm_lstm/strided_slice_2:output:0,model/lstm_lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/mul_1г
model/lstm_lstm/lstm_cell/mul_2Mul(model/lstm_lstm/strided_slice_2:output:0,model/lstm_lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/mul_2г
model/lstm_lstm/lstm_cell/mul_3Mul(model/lstm_lstm/strided_slice_2:output:0,model/lstm_lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/mul_3
model/lstm_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
model/lstm_lstm/lstm_cell/Const
)model/lstm_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model/lstm_lstm/lstm_cell/split/split_dimи
.model/lstm_lstm/lstm_cell/split/ReadVariableOpReadVariableOp7model_lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype020
.model/lstm_lstm/lstm_cell/split/ReadVariableOp
model/lstm_lstm/lstm_cell/splitSplit2model/lstm_lstm/lstm_cell/split/split_dim:output:06model/lstm_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2!
model/lstm_lstm/lstm_cell/splitЭ
 model/lstm_lstm/lstm_cell/MatMulMatMul!model/lstm_lstm/lstm_cell/mul:z:0(model/lstm_lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 model/lstm_lstm/lstm_cell/MatMulг
"model/lstm_lstm/lstm_cell/MatMul_1MatMul#model/lstm_lstm/lstm_cell/mul_1:z:0(model/lstm_lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2$
"model/lstm_lstm/lstm_cell/MatMul_1г
"model/lstm_lstm/lstm_cell/MatMul_2MatMul#model/lstm_lstm/lstm_cell/mul_2:z:0(model/lstm_lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2$
"model/lstm_lstm/lstm_cell/MatMul_2г
"model/lstm_lstm/lstm_cell/MatMul_3MatMul#model/lstm_lstm/lstm_cell/mul_3:z:0(model/lstm_lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2$
"model/lstm_lstm/lstm_cell/MatMul_3
!model/lstm_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2#
!model/lstm_lstm/lstm_cell/Const_1
+model/lstm_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model/lstm_lstm/lstm_cell/split_1/split_dimк
0model/lstm_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp9model_lstm_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype022
0model/lstm_lstm/lstm_cell/split_1/ReadVariableOp
!model/lstm_lstm/lstm_cell/split_1Split4model/lstm_lstm/lstm_cell/split_1/split_dim:output:08model/lstm_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2#
!model/lstm_lstm/lstm_cell/split_1л
!model/lstm_lstm/lstm_cell/BiasAddBiasAdd*model/lstm_lstm/lstm_cell/MatMul:product:0*model/lstm_lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!model/lstm_lstm/lstm_cell/BiasAddс
#model/lstm_lstm/lstm_cell/BiasAdd_1BiasAdd,model/lstm_lstm/lstm_cell/MatMul_1:product:0*model/lstm_lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2%
#model/lstm_lstm/lstm_cell/BiasAdd_1с
#model/lstm_lstm/lstm_cell/BiasAdd_2BiasAdd,model/lstm_lstm/lstm_cell/MatMul_2:product:0*model/lstm_lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2%
#model/lstm_lstm/lstm_cell/BiasAdd_2с
#model/lstm_lstm/lstm_cell/BiasAdd_3BiasAdd,model/lstm_lstm/lstm_cell/MatMul_3:product:0*model/lstm_lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2%
#model/lstm_lstm/lstm_cell/BiasAdd_3Ы
model/lstm_lstm/lstm_cell/mul_4Mulmodel/lstm_lstm/zeros:output:0.model/lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/mul_4Ы
model/lstm_lstm/lstm_cell/mul_5Mulmodel/lstm_lstm/zeros:output:0.model/lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/mul_5Ы
model/lstm_lstm/lstm_cell/mul_6Mulmodel/lstm_lstm/zeros:output:0.model/lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/mul_6Ы
model/lstm_lstm/lstm_cell/mul_7Mulmodel/lstm_lstm/zeros:output:0.model/lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/mul_7Ц
(model/lstm_lstm/lstm_cell/ReadVariableOpReadVariableOp1model_lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02*
(model/lstm_lstm/lstm_cell/ReadVariableOpЏ
-model/lstm_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-model/lstm_lstm/lstm_cell/strided_slice/stackГ
/model/lstm_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/model/lstm_lstm/lstm_cell/strided_slice/stack_1Г
/model/lstm_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/model/lstm_lstm/lstm_cell/strided_slice/stack_2
'model/lstm_lstm/lstm_cell/strided_sliceStridedSlice0model/lstm_lstm/lstm_cell/ReadVariableOp:value:06model/lstm_lstm/lstm_cell/strided_slice/stack:output:08model/lstm_lstm/lstm_cell/strided_slice/stack_1:output:08model/lstm_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2)
'model/lstm_lstm/lstm_cell/strided_sliceл
"model/lstm_lstm/lstm_cell/MatMul_4MatMul#model/lstm_lstm/lstm_cell/mul_4:z:00model/lstm_lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"model/lstm_lstm/lstm_cell/MatMul_4г
model/lstm_lstm/lstm_cell/addAddV2*model/lstm_lstm/lstm_cell/BiasAdd:output:0,model/lstm_lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/lstm_lstm/lstm_cell/addІ
!model/lstm_lstm/lstm_cell/SigmoidSigmoid!model/lstm_lstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!model/lstm_lstm/lstm_cell/SigmoidЪ
*model/lstm_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp1model_lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02,
*model/lstm_lstm/lstm_cell/ReadVariableOp_1Г
/model/lstm_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       21
/model/lstm_lstm/lstm_cell/strided_slice_1/stackЗ
1model/lstm_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1model/lstm_lstm/lstm_cell/strided_slice_1/stack_1З
1model/lstm_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1model/lstm_lstm/lstm_cell/strided_slice_1/stack_2Є
)model/lstm_lstm/lstm_cell/strided_slice_1StridedSlice2model/lstm_lstm/lstm_cell/ReadVariableOp_1:value:08model/lstm_lstm/lstm_cell/strided_slice_1/stack:output:0:model/lstm_lstm/lstm_cell/strided_slice_1/stack_1:output:0:model/lstm_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2+
)model/lstm_lstm/lstm_cell/strided_slice_1н
"model/lstm_lstm/lstm_cell/MatMul_5MatMul#model/lstm_lstm/lstm_cell/mul_5:z:02model/lstm_lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"model/lstm_lstm/lstm_cell/MatMul_5й
model/lstm_lstm/lstm_cell/add_1AddV2,model/lstm_lstm/lstm_cell/BiasAdd_1:output:0,model/lstm_lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/add_1Ќ
#model/lstm_lstm/lstm_cell/Sigmoid_1Sigmoid#model/lstm_lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#model/lstm_lstm/lstm_cell/Sigmoid_1Ц
model/lstm_lstm/lstm_cell/mul_8Mul'model/lstm_lstm/lstm_cell/Sigmoid_1:y:0 model/lstm_lstm/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/mul_8Ъ
*model/lstm_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp1model_lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02,
*model/lstm_lstm/lstm_cell/ReadVariableOp_2Г
/model/lstm_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/model/lstm_lstm/lstm_cell/strided_slice_2/stackЗ
1model/lstm_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   23
1model/lstm_lstm/lstm_cell/strided_slice_2/stack_1З
1model/lstm_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1model/lstm_lstm/lstm_cell/strided_slice_2/stack_2Є
)model/lstm_lstm/lstm_cell/strided_slice_2StridedSlice2model/lstm_lstm/lstm_cell/ReadVariableOp_2:value:08model/lstm_lstm/lstm_cell/strided_slice_2/stack:output:0:model/lstm_lstm/lstm_cell/strided_slice_2/stack_1:output:0:model/lstm_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2+
)model/lstm_lstm/lstm_cell/strided_slice_2н
"model/lstm_lstm/lstm_cell/MatMul_6MatMul#model/lstm_lstm/lstm_cell/mul_6:z:02model/lstm_lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"model/lstm_lstm/lstm_cell/MatMul_6й
model/lstm_lstm/lstm_cell/add_2AddV2,model/lstm_lstm/lstm_cell/BiasAdd_2:output:0,model/lstm_lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/add_2
model/lstm_lstm/lstm_cell/ReluRelu#model/lstm_lstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
model/lstm_lstm/lstm_cell/Reluа
model/lstm_lstm/lstm_cell/mul_9Mul%model/lstm_lstm/lstm_cell/Sigmoid:y:0,model/lstm_lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/mul_9Ч
model/lstm_lstm/lstm_cell/add_3AddV2#model/lstm_lstm/lstm_cell/mul_8:z:0#model/lstm_lstm/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/add_3Ъ
*model/lstm_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp1model_lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02,
*model/lstm_lstm/lstm_cell/ReadVariableOp_3Г
/model/lstm_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   21
/model/lstm_lstm/lstm_cell/strided_slice_3/stackЗ
1model/lstm_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1model/lstm_lstm/lstm_cell/strided_slice_3/stack_1З
1model/lstm_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1model/lstm_lstm/lstm_cell/strided_slice_3/stack_2Є
)model/lstm_lstm/lstm_cell/strided_slice_3StridedSlice2model/lstm_lstm/lstm_cell/ReadVariableOp_3:value:08model/lstm_lstm/lstm_cell/strided_slice_3/stack:output:0:model/lstm_lstm/lstm_cell/strided_slice_3/stack_1:output:0:model/lstm_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2+
)model/lstm_lstm/lstm_cell/strided_slice_3н
"model/lstm_lstm/lstm_cell/MatMul_7MatMul#model/lstm_lstm/lstm_cell/mul_7:z:02model/lstm_lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"model/lstm_lstm/lstm_cell/MatMul_7й
model/lstm_lstm/lstm_cell/add_4AddV2,model/lstm_lstm/lstm_cell/BiasAdd_3:output:0,model/lstm_lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
model/lstm_lstm/lstm_cell/add_4Ќ
#model/lstm_lstm/lstm_cell/Sigmoid_2Sigmoid#model/lstm_lstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#model/lstm_lstm/lstm_cell/Sigmoid_2Ѓ
 model/lstm_lstm/lstm_cell/Relu_1Relu#model/lstm_lstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 model/lstm_lstm/lstm_cell/Relu_1ж
 model/lstm_lstm/lstm_cell/mul_10Mul'model/lstm_lstm/lstm_cell/Sigmoid_2:y:0.model/lstm_lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 model/lstm_lstm/lstm_cell/mul_10Џ
-model/lstm_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2/
-model/lstm_lstm/TensorArrayV2_1/element_shapeј
model/lstm_lstm/TensorArrayV2_1TensorListReserve6model/lstm_lstm/TensorArrayV2_1/element_shape:output:0(model/lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model/lstm_lstm/TensorArrayV2_1n
model/lstm_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/lstm_lstm/time
(model/lstm_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(model/lstm_lstm/while/maximum_iterations
"model/lstm_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/lstm_lstm/while/loop_counterЫ
model/lstm_lstm/whileWhile+model/lstm_lstm/while/loop_counter:output:01model/lstm_lstm/while/maximum_iterations:output:0model/lstm_lstm/time:output:0(model/lstm_lstm/TensorArrayV2_1:handle:0model/lstm_lstm/zeros:output:0 model/lstm_lstm/zeros_1:output:0(model/lstm_lstm/strided_slice_1:output:0Gmodel/lstm_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07model_lstm_lstm_lstm_cell_split_readvariableop_resource9model_lstm_lstm_lstm_cell_split_1_readvariableop_resource1model_lstm_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*-
body%R#
!model_lstm_lstm_while_body_504779*-
cond%R#
!model_lstm_lstm_while_cond_504778*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
model/lstm_lstm/whileе
@model/lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2B
@model/lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shapeЈ
2model/lstm_lstm/TensorArrayV2Stack/TensorListStackTensorListStackmodel/lstm_lstm/while:output:3Imodel/lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype024
2model/lstm_lstm/TensorArrayV2Stack/TensorListStackЁ
%model/lstm_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%model/lstm_lstm/strided_slice_3/stack
'model/lstm_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'model/lstm_lstm/strided_slice_3/stack_1
'model/lstm_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/lstm_lstm/strided_slice_3/stack_2њ
model/lstm_lstm/strided_slice_3StridedSlice;model/lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0.model/lstm_lstm/strided_slice_3/stack:output:00model/lstm_lstm/strided_slice_3/stack_1:output:00model/lstm_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
model/lstm_lstm/strided_slice_3
 model/lstm_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model/lstm_lstm/transpose_1/permх
model/lstm_lstm/transpose_1	Transpose;model/lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0)model/lstm_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
model/lstm_lstm/transpose_1
model/lstm_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm_lstm/runtime{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
model/flatten/ConstЋ
model/flatten/ReshapeReshapemodel/lstm_lstm/transpose_1:y:0model/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
model/flatten/ReshapeС
&model/rnn_densef/MatMul/ReadVariableOpReadVariableOp/model_rnn_densef_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02(
&model/rnn_densef/MatMul/ReadVariableOpО
model/rnn_densef/MatMulMatMulmodel/flatten/Reshape:output:0.model/rnn_densef/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/rnn_densef/MatMulП
'model/rnn_densef/BiasAdd/ReadVariableOpReadVariableOp0model_rnn_densef_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model/rnn_densef/BiasAdd/ReadVariableOpХ
model/rnn_densef/BiasAddBiasAdd!model/rnn_densef/MatMul:product:0/model/rnn_densef/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/rnn_densef/BiasAdd
model/rnn_densef/SoftmaxSoftmax!model/rnn_densef/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model/rnn_densef/Softmax
IdentityIdentity"model/rnn_densef/Softmax:softmax:0^model/lstm_lstm/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::2.
model/lstm_lstm/whilemodel/lstm_lstm/while:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г
Ј
&__inference_model_layer_call_fn_507431

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5066262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
сЦ
а
while_body_505982
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resourceЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
lstm_cell/ones_like/ShapeShape*TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout/ConstЇ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeё
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2оўЙ20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 lstm_cell/dropout/GreaterEqual/yц
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/CastЂ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeї
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2кр22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_1/GreaterEqual/yю
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_1/GreaterEqualЃ
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/CastЊ
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeі
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЭЛ"22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_2/GreaterEqual/yю
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_2/GreaterEqualЃ
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/CastЊ
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeї
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2жнЏ22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_3/GreaterEqual/yю
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_3/GreaterEqualЃ
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/CastЊ
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul_1w
lstm_cell/ones_like_1/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_4/ConstЏ
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeї
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ћЩе22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_4/GreaterEqual/yю
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_4/GreaterEqualЃ
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/CastЊ
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_5/ConstЏ
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeї
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ја22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_5/GreaterEqual/yю
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_5/GreaterEqualЃ
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/CastЊ
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_6/ConstЏ
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeі
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2НН=22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_6/GreaterEqual/yю
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_6/GreaterEqualЃ
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/CastЊ
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_7/ConstЏ
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeї
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЇВЃ22
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_7/GreaterEqual/yю
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_7/GreaterEqualЃ
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/CastЊ
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul_1 
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mulІ
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1І
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2І
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЊ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЌ
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10Р
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3l

Identity_4Identitylstm_cell/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
е+

A__inference_model_layer_call_and_return_conditional_losses_506626

inputs
lstm_lstm_506596
lstm_lstm_506598
lstm_lstm_506600
rnn_densef_506604
rnn_densef_506606
identityЂ!lstm_lstm/StatefulPartitionedCallЂ"rnn_densef/StatefulPartitionedCall
!lstm_lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lstm_506596lstm_lstm_506598lstm_lstm_506600*
Tin
2*
Tout
2*+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_5061982#
!lstm_lstm/StatefulPartitionedCallд
flatten/PartitionedCallPartitionedCall*lstm_lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5065052
flatten/PartitionedCall
"rnn_densef/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0rnn_densef_506604rnn_densef_506606*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_rnn_densef_layer_call_and_return_conditional_losses_5065242$
"rnn_densef/StatefulPartitionedCallЧ
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_506596*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addл
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOplstm_lstm_506600*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addШ
IdentityIdentity+rnn_densef/StatefulPartitionedCall:output:0"^lstm_lstm/StatefulPartitionedCall#^rnn_densef/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::2F
!lstm_lstm/StatefulPartitionedCall!lstm_lstm/StatefulPartitionedCall2H
"rnn_densef/StatefulPartitionedCall"rnn_densef/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ї
$__inference_signature_wrapper_506728
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_5049242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

ћ
while_cond_506316
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_506316___redundant_placeholder0.
*while_cond_506316___redundant_placeholder1.
*while_cond_506316___redundant_placeholder2.
*while_cond_506316___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ФП
ђ
A__inference_model_layer_call_and_return_conditional_losses_507136

inputs5
1lstm_lstm_lstm_cell_split_readvariableop_resource7
3lstm_lstm_lstm_cell_split_1_readvariableop_resource/
+lstm_lstm_lstm_cell_readvariableop_resource-
)rnn_densef_matmul_readvariableop_resource.
*rnn_densef_biasadd_readvariableop_resource
identityЂlstm_lstm/whileX
lstm_lstm/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_lstm/Shape
lstm_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_lstm/strided_slice/stack
lstm_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_lstm/strided_slice/stack_1
lstm_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_lstm/strided_slice/stack_2
lstm_lstm/strided_sliceStridedSlicelstm_lstm/Shape:output:0&lstm_lstm/strided_slice/stack:output:0(lstm_lstm/strided_slice/stack_1:output:0(lstm_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_lstm/strided_slicep
lstm_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros/mul/y
lstm_lstm/zeros/mulMul lstm_lstm/strided_slice:output:0lstm_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros/muls
lstm_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_lstm/zeros/Less/y
lstm_lstm/zeros/LessLesslstm_lstm/zeros/mul:z:0lstm_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros/Lessv
lstm_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros/packed/1Ћ
lstm_lstm/zeros/packedPack lstm_lstm/strided_slice:output:0!lstm_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_lstm/zeros/packeds
lstm_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/zeros/Const
lstm_lstm/zerosFilllstm_lstm/zeros/packed:output:0lstm_lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/zerost
lstm_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros_1/mul/y
lstm_lstm/zeros_1/mulMul lstm_lstm/strided_slice:output:0 lstm_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros_1/mulw
lstm_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_lstm/zeros_1/Less/y
lstm_lstm/zeros_1/LessLesslstm_lstm/zeros_1/mul:z:0!lstm_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_lstm/zeros_1/Lessz
lstm_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/zeros_1/packed/1Б
lstm_lstm/zeros_1/packedPack lstm_lstm/strided_slice:output:0#lstm_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_lstm/zeros_1/packedw
lstm_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/zeros_1/ConstЅ
lstm_lstm/zeros_1Fill!lstm_lstm/zeros_1/packed:output:0 lstm_lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/zeros_1
lstm_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_lstm/transpose/perm
lstm_lstm/transpose	Transposeinputs!lstm_lstm/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_lstm/transposem
lstm_lstm/Shape_1Shapelstm_lstm/transpose:y:0*
T0*
_output_shapes
:2
lstm_lstm/Shape_1
lstm_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_lstm/strided_slice_1/stack
!lstm_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_1/stack_1
!lstm_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_1/stack_2Њ
lstm_lstm/strided_slice_1StridedSlicelstm_lstm/Shape_1:output:0(lstm_lstm/strided_slice_1/stack:output:0*lstm_lstm/strided_slice_1/stack_1:output:0*lstm_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_lstm/strided_slice_1
%lstm_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%lstm_lstm/TensorArrayV2/element_shapeк
lstm_lstm/TensorArrayV2TensorListReserve.lstm_lstm/TensorArrayV2/element_shape:output:0"lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_lstm/TensorArrayV2г
?lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape 
1lstm_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_lstm/transpose:y:0Hlstm_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1lstm_lstm/TensorArrayUnstack/TensorListFromTensor
lstm_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_lstm/strided_slice_2/stack
!lstm_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_2/stack_1
!lstm_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_2/stack_2И
lstm_lstm/strided_slice_2StridedSlicelstm_lstm/transpose:y:0(lstm_lstm/strided_slice_2/stack:output:0*lstm_lstm/strided_slice_2/stack_1:output:0*lstm_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_lstm/strided_slice_2
#lstm_lstm/lstm_cell/ones_like/ShapeShape"lstm_lstm/strided_slice_2:output:0*
T0*
_output_shapes
:2%
#lstm_lstm/lstm_cell/ones_like/Shape
#lstm_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_lstm/lstm_cell/ones_like/Constд
lstm_lstm/lstm_cell/ones_likeFill,lstm_lstm/lstm_cell/ones_like/Shape:output:0,lstm_lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/ones_like
!lstm_lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2#
!lstm_lstm/lstm_cell/dropout/ConstЯ
lstm_lstm/lstm_cell/dropout/MulMul&lstm_lstm/lstm_cell/ones_like:output:0*lstm_lstm/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
lstm_lstm/lstm_cell/dropout/Mul
!lstm_lstm/lstm_cell/dropout/ShapeShape&lstm_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2#
!lstm_lstm/lstm_cell/dropout/Shape
8lstm_lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform*lstm_lstm/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2дНЭ2:
8lstm_lstm/lstm_cell/dropout/random_uniform/RandomUniform
*lstm_lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2,
*lstm_lstm/lstm_cell/dropout/GreaterEqual/y
(lstm_lstm/lstm_cell/dropout/GreaterEqualGreaterEqualAlstm_lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:03lstm_lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(lstm_lstm/lstm_cell/dropout/GreaterEqualЛ
 lstm_lstm/lstm_cell/dropout/CastCast,lstm_lstm/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2"
 lstm_lstm/lstm_cell/dropout/CastЪ
!lstm_lstm/lstm_cell/dropout/Mul_1Mul#lstm_lstm/lstm_cell/dropout/Mul:z:0$lstm_lstm/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!lstm_lstm/lstm_cell/dropout/Mul_1
#lstm_lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2%
#lstm_lstm/lstm_cell/dropout_1/Constе
!lstm_lstm/lstm_cell/dropout_1/MulMul&lstm_lstm/lstm_cell/ones_like:output:0,lstm_lstm/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!lstm_lstm/lstm_cell/dropout_1/Mul 
#lstm_lstm/lstm_cell/dropout_1/ShapeShape&lstm_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2%
#lstm_lstm/lstm_cell/dropout_1/Shape
:lstm_lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform,lstm_lstm/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЯЕ2<
:lstm_lstm/lstm_cell/dropout_1/random_uniform/RandomUniformЁ
,lstm_lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_lstm/lstm_cell/dropout_1/GreaterEqual/y
*lstm_lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqualClstm_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:05lstm_lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*lstm_lstm/lstm_cell/dropout_1/GreaterEqualС
"lstm_lstm/lstm_cell/dropout_1/CastCast.lstm_lstm/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2$
"lstm_lstm/lstm_cell/dropout_1/Castв
#lstm_lstm/lstm_cell/dropout_1/Mul_1Mul%lstm_lstm/lstm_cell/dropout_1/Mul:z:0&lstm_lstm/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#lstm_lstm/lstm_cell/dropout_1/Mul_1
#lstm_lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2%
#lstm_lstm/lstm_cell/dropout_2/Constе
!lstm_lstm/lstm_cell/dropout_2/MulMul&lstm_lstm/lstm_cell/ones_like:output:0,lstm_lstm/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!lstm_lstm/lstm_cell/dropout_2/Mul 
#lstm_lstm/lstm_cell/dropout_2/ShapeShape&lstm_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2%
#lstm_lstm/lstm_cell/dropout_2/Shape
:lstm_lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform,lstm_lstm/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ьНС2<
:lstm_lstm/lstm_cell/dropout_2/random_uniform/RandomUniformЁ
,lstm_lstm/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_lstm/lstm_cell/dropout_2/GreaterEqual/y
*lstm_lstm/lstm_cell/dropout_2/GreaterEqualGreaterEqualClstm_lstm/lstm_cell/dropout_2/random_uniform/RandomUniform:output:05lstm_lstm/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*lstm_lstm/lstm_cell/dropout_2/GreaterEqualС
"lstm_lstm/lstm_cell/dropout_2/CastCast.lstm_lstm/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2$
"lstm_lstm/lstm_cell/dropout_2/Castв
#lstm_lstm/lstm_cell/dropout_2/Mul_1Mul%lstm_lstm/lstm_cell/dropout_2/Mul:z:0&lstm_lstm/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#lstm_lstm/lstm_cell/dropout_2/Mul_1
#lstm_lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2%
#lstm_lstm/lstm_cell/dropout_3/Constе
!lstm_lstm/lstm_cell/dropout_3/MulMul&lstm_lstm/lstm_cell/ones_like:output:0,lstm_lstm/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!lstm_lstm/lstm_cell/dropout_3/Mul 
#lstm_lstm/lstm_cell/dropout_3/ShapeShape&lstm_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2%
#lstm_lstm/lstm_cell/dropout_3/Shape
:lstm_lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform,lstm_lstm/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2бТм2<
:lstm_lstm/lstm_cell/dropout_3/random_uniform/RandomUniformЁ
,lstm_lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2.
,lstm_lstm/lstm_cell/dropout_3/GreaterEqual/y
*lstm_lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqualClstm_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:05lstm_lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*lstm_lstm/lstm_cell/dropout_3/GreaterEqualС
"lstm_lstm/lstm_cell/dropout_3/CastCast.lstm_lstm/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2$
"lstm_lstm/lstm_cell/dropout_3/Castв
#lstm_lstm/lstm_cell/dropout_3/Mul_1Mul%lstm_lstm/lstm_cell/dropout_3/Mul:z:0&lstm_lstm/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#lstm_lstm/lstm_cell/dropout_3/Mul_1
%lstm_lstm/lstm_cell/ones_like_1/ShapeShapelstm_lstm/zeros:output:0*
T0*
_output_shapes
:2'
%lstm_lstm/lstm_cell/ones_like_1/Shape
%lstm_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%lstm_lstm/lstm_cell/ones_like_1/Constм
lstm_lstm/lstm_cell/ones_like_1Fill.lstm_lstm/lstm_cell/ones_like_1/Shape:output:0.lstm_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
lstm_lstm/lstm_cell/ones_like_1
#lstm_lstm/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2%
#lstm_lstm/lstm_cell/dropout_4/Constз
!lstm_lstm/lstm_cell/dropout_4/MulMul(lstm_lstm/lstm_cell/ones_like_1:output:0,lstm_lstm/lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!lstm_lstm/lstm_cell/dropout_4/MulЂ
#lstm_lstm/lstm_cell/dropout_4/ShapeShape(lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_lstm/lstm_cell/dropout_4/Shape
:lstm_lstm/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform,lstm_lstm/lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2юЅ2<
:lstm_lstm/lstm_cell/dropout_4/random_uniform/RandomUniformЁ
,lstm_lstm/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72.
,lstm_lstm/lstm_cell/dropout_4/GreaterEqual/y
*lstm_lstm/lstm_cell/dropout_4/GreaterEqualGreaterEqualClstm_lstm/lstm_cell/dropout_4/random_uniform/RandomUniform:output:05lstm_lstm/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*lstm_lstm/lstm_cell/dropout_4/GreaterEqualС
"lstm_lstm/lstm_cell/dropout_4/CastCast.lstm_lstm/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2$
"lstm_lstm/lstm_cell/dropout_4/Castв
#lstm_lstm/lstm_cell/dropout_4/Mul_1Mul%lstm_lstm/lstm_cell/dropout_4/Mul:z:0&lstm_lstm/lstm_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#lstm_lstm/lstm_cell/dropout_4/Mul_1
#lstm_lstm/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2%
#lstm_lstm/lstm_cell/dropout_5/Constз
!lstm_lstm/lstm_cell/dropout_5/MulMul(lstm_lstm/lstm_cell/ones_like_1:output:0,lstm_lstm/lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!lstm_lstm/lstm_cell/dropout_5/MulЂ
#lstm_lstm/lstm_cell/dropout_5/ShapeShape(lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_lstm/lstm_cell/dropout_5/Shape
:lstm_lstm/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform,lstm_lstm/lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ДШ 2<
:lstm_lstm/lstm_cell/dropout_5/random_uniform/RandomUniformЁ
,lstm_lstm/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72.
,lstm_lstm/lstm_cell/dropout_5/GreaterEqual/y
*lstm_lstm/lstm_cell/dropout_5/GreaterEqualGreaterEqualClstm_lstm/lstm_cell/dropout_5/random_uniform/RandomUniform:output:05lstm_lstm/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*lstm_lstm/lstm_cell/dropout_5/GreaterEqualС
"lstm_lstm/lstm_cell/dropout_5/CastCast.lstm_lstm/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2$
"lstm_lstm/lstm_cell/dropout_5/Castв
#lstm_lstm/lstm_cell/dropout_5/Mul_1Mul%lstm_lstm/lstm_cell/dropout_5/Mul:z:0&lstm_lstm/lstm_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#lstm_lstm/lstm_cell/dropout_5/Mul_1
#lstm_lstm/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2%
#lstm_lstm/lstm_cell/dropout_6/Constз
!lstm_lstm/lstm_cell/dropout_6/MulMul(lstm_lstm/lstm_cell/ones_like_1:output:0,lstm_lstm/lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!lstm_lstm/lstm_cell/dropout_6/MulЂ
#lstm_lstm/lstm_cell/dropout_6/ShapeShape(lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_lstm/lstm_cell/dropout_6/Shape
:lstm_lstm/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform,lstm_lstm/lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Іе2<
:lstm_lstm/lstm_cell/dropout_6/random_uniform/RandomUniformЁ
,lstm_lstm/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72.
,lstm_lstm/lstm_cell/dropout_6/GreaterEqual/y
*lstm_lstm/lstm_cell/dropout_6/GreaterEqualGreaterEqualClstm_lstm/lstm_cell/dropout_6/random_uniform/RandomUniform:output:05lstm_lstm/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*lstm_lstm/lstm_cell/dropout_6/GreaterEqualС
"lstm_lstm/lstm_cell/dropout_6/CastCast.lstm_lstm/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2$
"lstm_lstm/lstm_cell/dropout_6/Castв
#lstm_lstm/lstm_cell/dropout_6/Mul_1Mul%lstm_lstm/lstm_cell/dropout_6/Mul:z:0&lstm_lstm/lstm_cell/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#lstm_lstm/lstm_cell/dropout_6/Mul_1
#lstm_lstm/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2%
#lstm_lstm/lstm_cell/dropout_7/Constз
!lstm_lstm/lstm_cell/dropout_7/MulMul(lstm_lstm/lstm_cell/ones_like_1:output:0,lstm_lstm/lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!lstm_lstm/lstm_cell/dropout_7/MulЂ
#lstm_lstm/lstm_cell/dropout_7/ShapeShape(lstm_lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2%
#lstm_lstm/lstm_cell/dropout_7/Shape
:lstm_lstm/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform,lstm_lstm/lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2па2<
:lstm_lstm/lstm_cell/dropout_7/random_uniform/RandomUniformЁ
,lstm_lstm/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72.
,lstm_lstm/lstm_cell/dropout_7/GreaterEqual/y
*lstm_lstm/lstm_cell/dropout_7/GreaterEqualGreaterEqualClstm_lstm/lstm_cell/dropout_7/random_uniform/RandomUniform:output:05lstm_lstm/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*lstm_lstm/lstm_cell/dropout_7/GreaterEqualС
"lstm_lstm/lstm_cell/dropout_7/CastCast.lstm_lstm/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2$
"lstm_lstm/lstm_cell/dropout_7/Castв
#lstm_lstm/lstm_cell/dropout_7/Mul_1Mul%lstm_lstm/lstm_cell/dropout_7/Mul:z:0&lstm_lstm/lstm_cell/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#lstm_lstm/lstm_cell/dropout_7/Mul_1Ж
lstm_lstm/lstm_cell/mulMul"lstm_lstm/strided_slice_2:output:0%lstm_lstm/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mulМ
lstm_lstm/lstm_cell/mul_1Mul"lstm_lstm/strided_slice_2:output:0'lstm_lstm/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_1М
lstm_lstm/lstm_cell/mul_2Mul"lstm_lstm/strided_slice_2:output:0'lstm_lstm/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_2М
lstm_lstm/lstm_cell/mul_3Mul"lstm_lstm/strided_slice_2:output:0'lstm_lstm/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_3x
lstm_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/lstm_cell/Const
#lstm_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_lstm/lstm_cell/split/split_dimЦ
(lstm_lstm/lstm_cell/split/ReadVariableOpReadVariableOp1lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02*
(lstm_lstm/lstm_cell/split/ReadVariableOpї
lstm_lstm/lstm_cell/splitSplit,lstm_lstm/lstm_cell/split/split_dim:output:00lstm_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_lstm/lstm_cell/splitЕ
lstm_lstm/lstm_cell/MatMulMatMullstm_lstm/lstm_cell/mul:z:0"lstm_lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMulЛ
lstm_lstm/lstm_cell/MatMul_1MatMullstm_lstm/lstm_cell/mul_1:z:0"lstm_lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_1Л
lstm_lstm/lstm_cell/MatMul_2MatMullstm_lstm/lstm_cell/mul_2:z:0"lstm_lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_2Л
lstm_lstm/lstm_cell/MatMul_3MatMullstm_lstm/lstm_cell/mul_3:z:0"lstm_lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_3|
lstm_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_lstm/lstm_cell/Const_1
%lstm_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%lstm_lstm/lstm_cell/split_1/split_dimШ
*lstm_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp3lstm_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02,
*lstm_lstm/lstm_cell/split_1/ReadVariableOpя
lstm_lstm/lstm_cell/split_1Split.lstm_lstm/lstm_cell/split_1/split_dim:output:02lstm_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_lstm/lstm_cell/split_1У
lstm_lstm/lstm_cell/BiasAddBiasAdd$lstm_lstm/lstm_cell/MatMul:product:0$lstm_lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/BiasAddЩ
lstm_lstm/lstm_cell/BiasAdd_1BiasAdd&lstm_lstm/lstm_cell/MatMul_1:product:0$lstm_lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/BiasAdd_1Щ
lstm_lstm/lstm_cell/BiasAdd_2BiasAdd&lstm_lstm/lstm_cell/MatMul_2:product:0$lstm_lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/BiasAdd_2Щ
lstm_lstm/lstm_cell/BiasAdd_3BiasAdd&lstm_lstm/lstm_cell/MatMul_3:product:0$lstm_lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/BiasAdd_3В
lstm_lstm/lstm_cell/mul_4Mullstm_lstm/zeros:output:0'lstm_lstm/lstm_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_4В
lstm_lstm/lstm_cell/mul_5Mullstm_lstm/zeros:output:0'lstm_lstm/lstm_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_5В
lstm_lstm/lstm_cell/mul_6Mullstm_lstm/zeros:output:0'lstm_lstm/lstm_cell/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_6В
lstm_lstm/lstm_cell/mul_7Mullstm_lstm/zeros:output:0'lstm_lstm/lstm_cell/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_7Д
"lstm_lstm/lstm_cell/ReadVariableOpReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02$
"lstm_lstm/lstm_cell/ReadVariableOpЃ
'lstm_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2)
'lstm_lstm/lstm_cell/strided_slice/stackЇ
)lstm_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)lstm_lstm/lstm_cell/strided_slice/stack_1Ї
)lstm_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)lstm_lstm/lstm_cell/strided_slice/stack_2є
!lstm_lstm/lstm_cell/strided_sliceStridedSlice*lstm_lstm/lstm_cell/ReadVariableOp:value:00lstm_lstm/lstm_cell/strided_slice/stack:output:02lstm_lstm/lstm_cell/strided_slice/stack_1:output:02lstm_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2#
!lstm_lstm/lstm_cell/strided_sliceУ
lstm_lstm/lstm_cell/MatMul_4MatMullstm_lstm/lstm_cell/mul_4:z:0*lstm_lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_4Л
lstm_lstm/lstm_cell/addAddV2$lstm_lstm/lstm_cell/BiasAdd:output:0&lstm_lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add
lstm_lstm/lstm_cell/SigmoidSigmoidlstm_lstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/SigmoidИ
$lstm_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_1Ї
)lstm_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2+
)lstm_lstm/lstm_cell/strided_slice_1/stackЋ
+lstm_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_lstm/lstm_cell/strided_slice_1/stack_1Ћ
+lstm_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_1/stack_2
#lstm_lstm/lstm_cell/strided_slice_1StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_1:value:02lstm_lstm/lstm_cell/strided_slice_1/stack:output:04lstm_lstm/lstm_cell/strided_slice_1/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_1Х
lstm_lstm/lstm_cell/MatMul_5MatMullstm_lstm/lstm_cell/mul_5:z:0,lstm_lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_5С
lstm_lstm/lstm_cell/add_1AddV2&lstm_lstm/lstm_cell/BiasAdd_1:output:0&lstm_lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add_1
lstm_lstm/lstm_cell/Sigmoid_1Sigmoidlstm_lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/Sigmoid_1Ў
lstm_lstm/lstm_cell/mul_8Mul!lstm_lstm/lstm_cell/Sigmoid_1:y:0lstm_lstm/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_8И
$lstm_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_2Ї
)lstm_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)lstm_lstm/lstm_cell/strided_slice_2/stackЋ
+lstm_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2-
+lstm_lstm/lstm_cell/strided_slice_2/stack_1Ћ
+lstm_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_2/stack_2
#lstm_lstm/lstm_cell/strided_slice_2StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_2:value:02lstm_lstm/lstm_cell/strided_slice_2/stack:output:04lstm_lstm/lstm_cell/strided_slice_2/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_2Х
lstm_lstm/lstm_cell/MatMul_6MatMullstm_lstm/lstm_cell/mul_6:z:0,lstm_lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_6С
lstm_lstm/lstm_cell/add_2AddV2&lstm_lstm/lstm_cell/BiasAdd_2:output:0&lstm_lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add_2
lstm_lstm/lstm_cell/ReluRelulstm_lstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/ReluИ
lstm_lstm/lstm_cell/mul_9Mullstm_lstm/lstm_cell/Sigmoid:y:0&lstm_lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_9Џ
lstm_lstm/lstm_cell/add_3AddV2lstm_lstm/lstm_cell/mul_8:z:0lstm_lstm/lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add_3И
$lstm_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02&
$lstm_lstm/lstm_cell/ReadVariableOp_3Ї
)lstm_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2+
)lstm_lstm/lstm_cell/strided_slice_3/stackЋ
+lstm_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+lstm_lstm/lstm_cell/strided_slice_3/stack_1Ћ
+lstm_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+lstm_lstm/lstm_cell/strided_slice_3/stack_2
#lstm_lstm/lstm_cell/strided_slice_3StridedSlice,lstm_lstm/lstm_cell/ReadVariableOp_3:value:02lstm_lstm/lstm_cell/strided_slice_3/stack:output:04lstm_lstm/lstm_cell/strided_slice_3/stack_1:output:04lstm_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2%
#lstm_lstm/lstm_cell/strided_slice_3Х
lstm_lstm/lstm_cell/MatMul_7MatMullstm_lstm/lstm_cell/mul_7:z:0,lstm_lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/MatMul_7С
lstm_lstm/lstm_cell/add_4AddV2&lstm_lstm/lstm_cell/BiasAdd_3:output:0&lstm_lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/add_4
lstm_lstm/lstm_cell/Sigmoid_2Sigmoidlstm_lstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/Sigmoid_2
lstm_lstm/lstm_cell/Relu_1Relulstm_lstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/Relu_1О
lstm_lstm/lstm_cell/mul_10Mul!lstm_lstm/lstm_cell/Sigmoid_2:y:0(lstm_lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_lstm/lstm_cell/mul_10Ѓ
'lstm_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2)
'lstm_lstm/TensorArrayV2_1/element_shapeр
lstm_lstm/TensorArrayV2_1TensorListReserve0lstm_lstm/TensorArrayV2_1/element_shape:output:0"lstm_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_lstm/TensorArrayV2_1b
lstm_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_lstm/time
"lstm_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"lstm_lstm/while/maximum_iterations~
lstm_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_lstm/while/loop_counterё
lstm_lstm/whileWhile%lstm_lstm/while/loop_counter:output:0+lstm_lstm/while/maximum_iterations:output:0lstm_lstm/time:output:0"lstm_lstm/TensorArrayV2_1:handle:0lstm_lstm/zeros:output:0lstm_lstm/zeros_1:output:0"lstm_lstm/strided_slice_1:output:0Alstm_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_lstm_lstm_cell_split_readvariableop_resource3lstm_lstm_lstm_cell_split_1_readvariableop_resource+lstm_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_lstm_while_body_506911*'
condR
lstm_lstm_while_cond_506910*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
lstm_lstm/whileЩ
:lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2<
:lstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape
,lstm_lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm_lstm/while:output:3Clstm_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02.
,lstm_lstm/TensorArrayV2Stack/TensorListStack
lstm_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2!
lstm_lstm/strided_slice_3/stack
!lstm_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!lstm_lstm/strided_slice_3/stack_1
!lstm_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!lstm_lstm/strided_slice_3/stack_2ж
lstm_lstm/strided_slice_3StridedSlice5lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0(lstm_lstm/strided_slice_3/stack:output:0*lstm_lstm/strided_slice_3/stack_1:output:0*lstm_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_lstm/strided_slice_3
lstm_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_lstm/transpose_1/permЭ
lstm_lstm/transpose_1	Transpose5lstm_lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm_lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
lstm_lstm/transpose_1z
lstm_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_lstm/runtimeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten/Const
flatten/ReshapeReshapelstm_lstm/transpose_1:y:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/ReshapeЏ
 rnn_densef/MatMul/ReadVariableOpReadVariableOp)rnn_densef_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02"
 rnn_densef/MatMul/ReadVariableOpІ
rnn_densef/MatMulMatMulflatten/Reshape:output:0(rnn_densef/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
rnn_densef/MatMul­
!rnn_densef/BiasAdd/ReadVariableOpReadVariableOp*rnn_densef_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!rnn_densef/BiasAdd/ReadVariableOp­
rnn_densef/BiasAddBiasAddrnn_densef/MatMul:product:0)rnn_densef/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
rnn_densef/BiasAdd
rnn_densef/SoftmaxSoftmaxrnn_densef/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
rnn_densef/Softmaxш
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp1lstm_lstm_lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addі
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+lstm_lstm_lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add
IdentityIdentityrnn_densef/Softmax:softmax:0^lstm_lstm/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ:::::2"
lstm_lstm/whilelstm_lstm/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ю

__inference_loss_fn_1_509217P
Llstm_lstm_lstm_cell_recurrent_kernel_regularizer_abs_readvariableop_resource
identity
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpLlstm_lstm_lstm_cell_recurrent_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add{
IdentityIdentity8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ьd

E__inference_lstm_cell_layer_call_and_return_conditional_losses_509157

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02
split/ReadVariableOpЇ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
splite
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMulk
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_1k
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_2k
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype02
split_1/ReadVariableOp
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
	BiasAdd_3g
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_4g
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_5g
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_6g
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_7x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ќ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_1`
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_8|
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Reluh
mul_9MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_9_
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_3|
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relu_1n
mul_10MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_10д
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addт
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add^
IdentityIdentity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityb

Identity_1Identity
mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1a

Identity_2Identity	add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ы
я
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_507861

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identityЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout/ConstЇ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shape№
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2щ[20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 lstm_cell/dropout/GreaterEqual/yц
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/CastЂ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeї
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed222
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_1/GreaterEqual/yю
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_1/GreaterEqualЃ
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/CastЊ
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeї
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2зЫЇ22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_2/GreaterEqual/yю
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_2/GreaterEqualЃ
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/CastЊ
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeі
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2j22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_3/GreaterEqual/yю
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_3/GreaterEqualЃ
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/CastЊ
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul_1x
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_4/ConstЏ
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeї
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЂуЃ22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_4/GreaterEqual/yю
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_4/GreaterEqualЃ
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/CastЊ
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_5/ConstЏ
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeї
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2 ђЧ22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_5/GreaterEqual/yю
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_5/GreaterEqualЃ
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/CastЊ
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_6/ConstЏ
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeї
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2њт22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_6/GreaterEqual/yю
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_6/GreaterEqualЃ
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/CastЊ
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_7/ConstЏ
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeї
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2щ22
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_7/GreaterEqual/yю
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_7/GreaterEqualЃ
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/CastЊ
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЈ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЊ
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterл
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_507645*
condR
while_cond_507644*K
output_shapes:
8: : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeо
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource*
_output_shapes

:@*
dtype02;
9lstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOpЫ
*lstm_lstm/lstm_cell/kernel/Regularizer/AbsAbsAlstm_lstm/lstm_cell/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2,
*lstm_lstm/lstm_cell/kernel/Regularizer/Abs­
,lstm_lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,lstm_lstm/lstm_cell/kernel/Regularizer/Constч
*lstm_lstm/lstm_cell/kernel/Regularizer/SumSum.lstm_lstm/lstm_cell/kernel/Regularizer/Abs:y:05lstm_lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/SumЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Зб82.
,lstm_lstm/lstm_cell/kernel/Regularizer/mul/xь
*lstm_lstm/lstm_cell/kernel/Regularizer/mulMul5lstm_lstm/lstm_cell/kernel/Regularizer/mul/x:output:03lstm_lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/mulЁ
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lstm_lstm/lstm_cell/kernel/Regularizer/add/xщ
*lstm_lstm/lstm_cell/kernel/Regularizer/addAddV25lstm_lstm/lstm_cell/kernel/Regularizer/add/x:output:0.lstm_lstm/lstm_cell/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*lstm_lstm/lstm_cell/kernel/Regularizer/addь
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes

:@*
dtype02E
Clstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOpщ
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsAbsKlstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/AbsС
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum8lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Abs:y:0?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/SumЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mulЕ
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addAddV2?lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/add/x:output:08lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 26
4lstm_lstm/lstm_cell/recurrent_kernel/Regularizer/addo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
тЦ
а
while_body_507645
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resourceЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
lstm_cell/ones_like/ShapeShape*TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout/ConstЇ
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeё
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ы20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 lstm_cell/dropout/GreaterEqual/yц
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/CastЂ
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_1/Const­
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeї
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЧюС22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_1/GreaterEqual/yю
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_1/GreaterEqualЃ
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/CastЊ
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_2/Const­
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeі
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ц<22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_2/GreaterEqual/yю
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_2/GreaterEqualЃ
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/CastЊ
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Э ?2
lstm_cell/dropout_3/Const­
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeї
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2 Г22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2$
"lstm_cell/dropout_3/GreaterEqual/yю
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_3/GreaterEqualЃ
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/CastЊ
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_3/Mul_1w
lstm_cell/ones_like_1/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_4/ConstЏ
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeї
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЏИ22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_4/GreaterEqual/yю
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_4/GreaterEqualЃ
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/CastЊ
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_5/ConstЏ
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeї
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2Нмф22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_5/GreaterEqual/yю
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_5/GreaterEqualЃ
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/CastЊ
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_6/ConstЏ
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeї
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2зэќ22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_6/GreaterEqual/yю
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_6/GreaterEqualЃ
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/CastЊ
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *T ?2
lstm_cell/dropout_7/ConstЏ
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeї
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЄЦч22
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72$
"lstm_cell/dropout_7/GreaterEqual/yю
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 lstm_cell/dropout_7/GreaterEqualЃ
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/CastЊ
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/dropout_7/Mul_1 
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mulІ
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1І
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2І
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЊ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЌ
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulplaceholder_2lstm_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulplaceholder_2lstm_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulplaceholder_2lstm_cell/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulplaceholder_2lstm_cell/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10Р
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3l

Identity_4Identitylstm_cell/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 

ћ
while_cond_505981
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_505981___redundant_placeholder0.
*while_cond_505981___redundant_placeholder1.
*while_cond_505981___redundant_placeholder2.
*while_cond_505981___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:

ћ
while_cond_505701
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1.
*while_cond_505701___redundant_placeholder0.
*while_cond_505701___redundant_placeholder1.
*while_cond_505701___redundant_placeholder2.
*while_cond_505701___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ч

*__inference_lstm_lstm_layer_call_fn_508154

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_5064692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ђ
Ў
F__inference_rnn_densef_layer_call_and_return_conditional_losses_508868

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
o
а
while_body_506317
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
)lstm_cell_split_readvariableop_resource_0/
+lstm_cell_split_1_readvariableop_resource_0'
#lstm_cell_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resourceЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
lstm_cell/ones_like/ShapeShape*TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/ConstЌ
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_likew
lstm_cell/ones_like_1/ShapeShapeplaceholder_2*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/ConstД
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/ones_like_1Ё
lstm_cell/mulMul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mulЅ
lstm_cell/mul_1Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_1Ѕ
lstm_cell/mul_2Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_2Ѕ
lstm_cell/mul_3Mul*TensorArrayV2Read/TensorListGetItem:item:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimЊ
lstm_cell/split/ReadVariableOpReadVariableOp)lstm_cell_split_readvariableop_resource_0*
_output_shapes

:@*
dtype02 
lstm_cell/split/ReadVariableOpЯ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dimЌ
 lstm_cell/split_1/ReadVariableOpReadVariableOp+lstm_cell_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype02"
 lstm_cell/split_1/ReadVariableOpЧ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAddЁ
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_1Ё
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_2Ё
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_4
lstm_cell/mul_5Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_5
lstm_cell/mul_6Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_6
lstm_cell/mul_7Mulplaceholder_2lstm_cell/ones_like_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2И
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/addv
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ф
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_1|
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ф
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_2o
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp#lstm_cell_readvariableop_resource_0*
_output_shapes

:@*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ф
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/add_4|
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/Relu_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
lstm_cell/mul_10Р
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholderlstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity_

Identity_1Identitywhile_maximum_iterations*
T0*
_output_shapes
: 2

Identity_1N

Identity_2Identityadd:z:0*
T0*
_output_shapes
: 2

Identity_2{

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2

Identity_3l

Identity_4Identitylstm_cell/mul_10:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_4k

Identity_5Identitylstm_cell/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"H
!lstm_cell_readvariableop_resource#lstm_cell_readvariableop_resource_0"X
)lstm_cell_split_1_readvariableop_resource+lstm_cell_split_1_readvariableop_resource_0"T
'lstm_cell_split_readvariableop_resource)lstm_cell_split_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ:џџџџџџџџџ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: "ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Б
serving_default
?
input_14
serving_default_input_1:0џџџџџџџџџ>

rnn_densef0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:тН
Ш*
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
V__call__
*W&call_and_return_all_conditional_losses
X_default_save_signature"(
_tf_keras_modelћ'{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "recurrent_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.001, "recurrent_dropout": 1e-05, "implementation": 1}, "name": "lstm_lstm", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["lstm_lstm", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "rnn_densef", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rnn_densef", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["rnn_densef", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 6]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "recurrent_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.001, "recurrent_dropout": 1e-05, "implementation": 1}, "name": "lstm_lstm", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["lstm_lstm", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "rnn_densef", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "rnn_densef", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["rnn_densef", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ё"ю
_tf_keras_input_layerЮ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ъ
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"С
_tf_keras_rnn_layerЃ{"class_name": "LSTM", "name": "lstm_lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "lstm_lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "recurrent_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.001, "recurrent_dropout": 1e-05, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 6]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 6]}}
П
trainable_variables
regularization_losses
	variables
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"А
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"я
_tf_keras_layerе{"class_name": "Dense", "name": "rnn_densef", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "rnn_densef", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 320}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 320]}}
­
iter

beta_1

beta_2
	decay
learning_ratemLmM mN!mO"mPvQvR vS!vT"vU"
	optimizer
C
 0
!1
"2
3
4"
trackable_list_wrapper
C
 0
!1
"2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
	variables
#layer_metrics
trainable_variables
$non_trainable_variables
%layer_regularization_losses
regularization_losses

&layers
'metrics
V__call__
X_default_save_signature
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
,
_serving_default"
signature_map
Ю	

 kernel
!recurrent_kernel
"bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layerљ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-05, "l2": 0.0}}, "recurrent_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0010000000474974513, "l2": 0.0}}, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.001, "recurrent_dropout": 1e-05, "implementation": 1}}
 "
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
Й
	variables
,layer_metrics
trainable_variables

-states
.non_trainable_variables
/layer_regularization_losses
regularization_losses

0layers
1metrics
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
2layer_metrics
trainable_variables
regularization_losses
3non_trainable_variables
4layer_regularization_losses
	variables

5layers
6metrics
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
$:"	Р2rnn_densef/kernel
:2rnn_densef/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
7layer_metrics
trainable_variables
regularization_losses
8non_trainable_variables
9layer_regularization_losses
	variables

:layers
;metrics
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*@2lstm_lstm/lstm_cell/kernel
6:4@2$lstm_lstm/lstm_cell/recurrent_kernel
&:$@2lstm_lstm/lstm_cell/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
­
>layer_metrics
(trainable_variables
)regularization_losses
?non_trainable_variables
@layer_regularization_losses
*	variables

Alayers
Bmetrics
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Л
	Ctotal
	Dcount
E	variables
F	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
џ
	Gtotal
	Hcount
I
_fn_kwargs
J	variables
K	keras_api"И
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
C0
D1"
trackable_list_wrapper
-
E	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
-
J	variables"
_generic_user_object
):'	Р2Adam/rnn_densef/kernel/m
": 2Adam/rnn_densef/bias/m
1:/@2!Adam/lstm_lstm/lstm_cell/kernel/m
;:9@2+Adam/lstm_lstm/lstm_cell/recurrent_kernel/m
+:)@2Adam/lstm_lstm/lstm_cell/bias/m
):'	Р2Adam/rnn_densef/kernel/v
": 2Adam/rnn_densef/bias/v
1:/@2!Adam/lstm_lstm/lstm_cell/kernel/v
;:9@2+Adam/lstm_lstm/lstm_cell/recurrent_kernel/v
+:)@2Adam/lstm_lstm/lstm_cell/bias/v
ц2у
&__inference_model_layer_call_fn_507431
&__inference_model_layer_call_fn_507446
&__inference_model_layer_call_fn_506639
&__inference_model_layer_call_fn_506687Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
A__inference_model_layer_call_and_return_conditional_losses_506590
A__inference_model_layer_call_and_return_conditional_losses_506557
A__inference_model_layer_call_and_return_conditional_losses_507136
A__inference_model_layer_call_and_return_conditional_losses_507416Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
у2р
!__inference__wrapped_model_504924К
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_1џџџџџџџџџ
2
*__inference_lstm_lstm_layer_call_fn_508835
*__inference_lstm_lstm_layer_call_fn_508143
*__inference_lstm_lstm_layer_call_fn_508154
*__inference_lstm_lstm_layer_call_fn_508846е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ї2є
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_508553
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_508824
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_507861
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_508132е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
(__inference_flatten_layer_call_fn_508857Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_508852Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_rnn_densef_layer_call_fn_508877Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_rnn_densef_layer_call_and_return_conditional_losses_508868Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
3B1
$__inference_signature_wrapper_506728input_1
2
*__inference_lstm_cell_layer_call_fn_509174
*__inference_lstm_cell_layer_call_fn_509191О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
E__inference_lstm_cell_layer_call_and_return_conditional_losses_509057
E__inference_lstm_cell_layer_call_and_return_conditional_losses_509157О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Г2А
__inference_loss_fn_0_509204
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Г2А
__inference_loss_fn_1_509217
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
!__inference__wrapped_model_504924v "!4Ђ1
*Ђ'
%"
input_1џџџџџџџџџ
Њ "7Њ4
2

rnn_densef$!

rnn_densefџџџџџџџџџЄ
C__inference_flatten_layer_call_and_return_conditional_losses_508852]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџР
 |
(__inference_flatten_layer_call_fn_508857P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџР;
__inference_loss_fn_0_509204 Ђ

Ђ 
Њ " ;
__inference_loss_fn_1_509217!Ђ

Ђ 
Њ " Ч
E__inference_lstm_cell_layer_call_and_return_conditional_losses_509057§ "!Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p
Њ "sЂp
iЂf

0/0џџџџџџџџџ
EB

0/1/0џџџџџџџџџ

0/1/1џџџџџџџџџ
 Ч
E__inference_lstm_cell_layer_call_and_return_conditional_losses_509157§ "!Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ
EB

0/1/0џџџџџџџџџ

0/1/1џџџџџџџџџ
 
*__inference_lstm_cell_layer_call_fn_509174э "!Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p
Њ "cЂ`

0џџџџџџџџџ
A>

1/0џџџџџџџџџ

1/1џџџџџџџџџ
*__inference_lstm_cell_layer_call_fn_509191э "!Ђ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџ
"
states/1џџџџџџџџџ
p 
Њ "cЂ`

0џџџџџџџџџ
A>

1/0џџџџџџџџџ

1/1џџџџџџџџџК
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_507861q "!?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ ")Ђ&

0џџџџџџџџџ
 К
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_508132q "!?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 д
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_508553 "!OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 д
E__inference_lstm_lstm_layer_call_and_return_conditional_losses_508824 "!OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 
*__inference_lstm_lstm_layer_call_fn_508143d "!?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
*__inference_lstm_lstm_layer_call_fn_508154d "!?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџЋ
*__inference_lstm_lstm_layer_call_fn_508835} "!OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџЋ
*__inference_lstm_lstm_layer_call_fn_508846} "!OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџБ
A__inference_model_layer_call_and_return_conditional_losses_506557l "!<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Б
A__inference_model_layer_call_and_return_conditional_losses_506590l "!<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 А
A__inference_model_layer_call_and_return_conditional_losses_507136k "!;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 А
A__inference_model_layer_call_and_return_conditional_losses_507416k "!;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
&__inference_model_layer_call_fn_506639_ "!<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
&__inference_model_layer_call_fn_506687_ "!<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
&__inference_model_layer_call_fn_507431^ "!;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
&__inference_model_layer_call_fn_507446^ "!;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЇ
F__inference_rnn_densef_layer_call_and_return_conditional_losses_508868]0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_rnn_densef_layer_call_fn_508877P0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "џџџџџџџџџЊ
$__inference_signature_wrapper_506728 "!?Ђ<
Ђ 
5Њ2
0
input_1%"
input_1џџџџџџџџџ"7Њ4
2

rnn_densef$!

rnn_densefџџџџџџџџџ