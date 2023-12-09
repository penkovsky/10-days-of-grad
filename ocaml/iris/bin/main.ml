open Lacaml.D

(* New weights *)
let newW (nin, nout) =
  let k = sqrt (1.0 /. float_of_int nin) in
  let w = Mat.random nin nout in
  Mat.map (fun x -> k *. x) w

(* Transformations *)
let loss y tgt =
  let diff = Mat.sub y tgt in
  Mat.(sum (map (fun x -> x *. x) diff))

let sigmoid x = Mat.map (fun x -> 1.0 /. (1.0 +. exp (-.x))) x

(* Their gradients *)
let sigmoid' x dY =
  let y = sigmoid x in
  let ones = Mat.make (Mat.dim1 y) (Mat.dim2 y) 1.0 in
  Mat.mul dY (Mat.mul y (Mat.sub ones y))

let linear' x dy =
  let m = float_of_int (Mat.dim1 x) in
  Mat.map (fun x -> x /. m) (gemm (Mat.transpose_copy x) dy)

let loss' y tgt =
  let diff = Mat.sub y tgt in
  Mat.map (fun x -> 2.0 *. x) diff

(* Building NN *)
let forward x w1 =
  let h = gemm x w1 in
  let y = sigmoid h in
  (h, y)

let descend gradF iterN gamma x0 =
  let step x =
    let gr = gradF x in
    Mat.scal gamma gr;
    Mat.sub x gr
  in
  let foo = ref x0 in
  for _ = 1 to iterN do
    foo := step !foo
  done;
  !foo

let grad (x, y) w1 =
  let h, y_pred = forward x w1 in
  let dE = loss' y_pred y in
  let dY = sigmoid' h dE in
  linear' x dY

(* Load data from a text file into a matrix *)
let load_matrix filename =
  let csv_data = Csv.load ~separator:' ' filename in
  let nrows = List.length csv_data in
  let ncols = List.length (List.hd csv_data) in
  let data = Array.make_matrix nrows ncols 0.0 in
  List.iteri
    (fun i row ->
      List.iteri (fun j value -> data.(i).(j) <- float_of_string value) row)
    csv_data;
  Mat.of_array data

(* Prints first five rows *)
let print5 matrix =
  let cols = Mat.dim2 matrix in
  for i = 1 to 5 do
    for j = 1 to cols do
      Printf.printf "%f " matrix.{i, j}
    done;
    print_newline ()
  done

let () =
  let dta = load_matrix "iris_x.dat" in
  let tgt = load_matrix "iris_y.dat" in

  let nin, nout = (4, 3) in

  let w1_rand = newW (nin, nout) in

  let epochs = 500 in
  let lr = 0.01 in

  let w1 = descend (grad (dta, tgt)) epochs lr w1_rand in

  let _, y_pred0 = forward dta w1_rand in
  let _, y_pred = forward dta w1 in

  Printf.printf "Initial loss: %f\n" (loss y_pred0 tgt);
  Printf.printf "Loss after training: %f\n" (loss y_pred tgt);

  print_endline "Data";
  print5 dta;

  print_endline "Targets";
  print5 tgt;

  print_endline "Some predictions by an untrained network:";
  print5 y_pred0;

  print_endline "Some predictions by a trained network:";
  print5 y_pred
