;(load "./chipotle/targets.lisp")
;(load "./chipotle/intrinsify.lisp")
;(load "./chipotle/vectorize.lisp")

(in-package :chipotle)

;;; ==============================================
;;;                    features
;;; ==============================================

;;; sse/avx vectorization
;;; ==============================================

(define-feature vectorize (input output &body body))

(implement vectorize (sse)
  `(vectorize# :sse ,input ,output ,@body))

(implement vectorize (avx)
  `(vectorize# :avx ,input ,output ,@body))


;;; SIMD width
;;; ==============================================

(define-feature simd-w ())
(implement simd-w (default) 1)
(implement simd-w (sse) 4)
(implement simd-w (avx) 8)
(implement simd-w (cuda) 16)


;;; SIMD height
;;; ==============================================

(define-feature simd-h ())
(implement simd-h (default) 1)
(implement simd-h (cuda) 8)


;;; image array data
;;; ==============================================

(define-feature array-type ())

(implement array-type (unsigned-char)
  ''uchar4*)

(implement array-type (floating-point)
  ''float4*)


;;; function setup
;;; ==============================================

(define-feature function-setup (name (&key (input '()) (output '())) &body body))


;;; setup cuda launch function and cuda kernel
(implement function-setup (cuda)
  (let* ((arrays (append input output))
	 (args (loop for array in arrays collect `(cast (array-type) (oref ,array data))))
	 (params (loop for array in input collect `(const (array-type) ,array)))
	 (params2 (loop for array in output collect `((array-type) ,array)))
	 (params+wh (append params params2 '((unsigned int w) (unsigned int h))))
	 (kernel (symbol-append 'kernel- name)))
    `(progn
       (comment "__launch_bounds__(16*8)" :prefix "")
       (function ,kernel ,params+wh -> (__global__ void)
	   (decl ((const int x = (+ (* blockIdx.x blockDim.x) threadIdx.x))
		  (const int y = (+ (* blockIdx.y blockDim.y) threadIdx.y)))
	       (if (or (>= x w) (>= y h))
		   (return))
	       ,@body))
       (function ,name ,(loop for array in arrays collect `(ub_image_cuda& ,array)) -> void
	   (decl ((const unsigned int w = (oref ,(first input) w))
		  (const unsigned int h = (oref ,(first input) h))		   
		  (dim3 threads = (dim3 (simd-w) (simd-h) 1))
		  (dim3 blocks = (dim3 (+ (/ w threads.x) (? (% w threads.x) 1 0))
				       (+ (/ h threads.y) (? (% h threads.y) 1 0))
				       1)))
	       (checked-cuda ,(format nil "image iteration ~a" kernel)
		 (launch ,kernel (:blocks blocks :threads threads) ,@args w h)))))))


;;; setup host functions for cpu, sse, avx
(implement function-setup (default)
  (let* ((arrays (append input output))
	 (args (loop for array in arrays collect `(cast (array-type) (oref ,array data))))
	 (params (loop for array in input collect `(const (array-type) ,array)))
	 (params2 (loop for array in output collect `((array-type) ,array)))
	 (params+wh (append params params2 '((unsigned int w) (unsigned int h))))
	 (filt (symbol-append 'filt- name)))
    `(progn
       (function ,filt ,params+wh -> void
	   ,@body)
       (function ,name ,(loop for array in arrays collect `(ub_image& ,array)) -> void
	   (funcall ,filt ,@args (oref ,(first input) 'w) (oref ,(first input) 'h))))))


;;; default loop, used for plain cpu implementation
;;; ==============================================

(define-feature loop-setup (&body body))

;;; dafault, do nothing / active for cuda
(implement loop-setup (default)
  `(progn ,@body))

;;; sse (+avx), do nothing 
(implement loop-setup (sse)
  `(progn ,@body))

;;; plain cpu, setup image iteration
(implement loop-setup (cpu)
  `(progn
     (comment "pragma omp parallel for collapse(2)" :prefix "#")
     (for ((int y = 0) (< y h) ++y)
	 (for ((int x = 0) (< x w) ++x)
	     ,@body))))


;;; absolute image array access
;;; ==============================================

(define-feature abs-array (array))

;;; x-y style access
(implement abs-array (default)
  `(aref ,array (cms-cuda::+ (cms-cuda::* y w) x)))

;;; linear i style acces
(implement abs-array (cpu point-operator)
  `(aref ,array i))


;;; border types, clamp (default) and exceed
;;; ==============================================

(define-feature border-type (dx dy checks condition &body body))

(defmacro array-pattern (a ra)
  `(cl::cond
     ((cl:= ,ra 0) ,a)
     ((cl:> ,ra 0) (cms-cuda::+ ,a ,ra))
     ((cl:< ,ra 0) (cms-cuda::- ,a (cl:- ,ra)))))

;;; array access with x,y offsets
;(defmacro rel-array (&rest rest)
  ;`(comment "nothing here"))
(defmacro rel-array (array rx ry)
  `(aref ,array (cms-cuda::+
		 (cms-cuda::* (array-pattern 'y ,ry) w)
		 (array-pattern 'x ,rx))))

;;; override relative array acces for mirror/repeat
(defmacro rel-override (array rx ry x-off y-off)
  `(aref ,array (cms-cuda::+
		 (cms-cuda::*
		   ,(cl:if y-off
			   `(cms-cuda::+ (array-pattern 'y ,ry) ,y-off)
			   `(array-pattern 'y ,ry))
		   w)
		 ,(cl:if x-off
			 `(cms-cuda::+ (array-pattern 'x ,rx) ,x-off)
			 `(array-pattern 'x ,rx)))))

;;; default: clamp out of bounds access
(implement border-type (default)
  (cl::cond ((cl:>  (cl:length condition) 1)
	     `(cms-cuda::if (&& ,@condition) ,@body))
	    ((cl:eq (cl:length condition) 1)
	     `(cms-cuda::if ,@condition ,@body))
	    (t `(progn ,@body))))

;;; exeed: prepare mirror and repeat access modes
(implement border-type (exceed)
  (let ((x-cond (first condition))
	(y-cond (cl:if (second condition)
		       (second condition)
		       (first condition))))
    (let ((x-off (cl:cond 
		   ((first checks)  `(border-left  ,x-cond))
		   ((second checks) `(border-right ,x-cond))
		   (t nil)))

	  (y-off (cl:cond 
		   ((fourth checks) `(border-bottom ,y-cond))
		   ((third checks)  `(border-top    ,y-cond))
		   (t nil))))
      (cl:if condition
	     `(macrolet ((rel-array (array rx ry)
			   `(rel-override ,array ,rx ,ry ,',x-off ,',y-off)))
		,@body)
	     `(progn ,@body)))))


;;; mirror and repeat out-of-bounds access
;;; ==============================================

(define-feature border-left   (cond))
(define-feature border-right  (cond))
(define-feature border-top    (cond))
(define-feature border-bottom (cond))

(implement border-left   (mirror) `(cms-cuda::* -2 (cms-cuda::not ,cond) (array-pattern 'x rx)))
(implement border-bottom (mirror) `(cms-cuda::* -2 (cms-cuda::not ,cond) (array-pattern 'y ry)))
(implement border-right  (mirror) `(cms-cuda::* -2 (cms-cuda::not ,cond) (cms-cuda::- (array-pattern 'x rx) w)))
(implement border-top    (mirror) `(cms-cuda::* -2 (cms-cuda::not ,cond) (cms-cuda::- (array-pattern 'y ry) h)))

(implement border-left   (repeat) `(cms-cuda::* (cms-cuda::not ,cond) w))
(implement border-bottom (repeat) `(cms-cuda::* (cms-cuda::not ,cond) h))
(implement border-right  (repeat) `(cms-cuda::* (cms-cuda::not ,cond) (cms-cuda::- w)))
(implement border-top    (repeat) `(cms-cuda::* (cms-cuda::not ,cond) (cms-cuda::- h)))


;;; goto and labels
;;; ==============================================

;;; cpu,sse,avx: no implementation needed
;;; standard implementation emits nothing

(define-feature goto-block (what))

;;; emit goto for cuda
(implement goto-block (cuda)
  `(goto ,what))

(define-feature block-label (what))

;;; emit label for cuda
(implement block-label (cuda)
  `(clabel ,what))


;;; general border handling macros
;;; ==============================================

;;; generate checks (clamp, exceed) for relative pixel acces
(defmacro boundary-checked-access (dx dy active-bounds &body body)
  (let ((check-left   (cl:and (cl:< dx 0) (find 'L active-bounds)))
	(check-right  (cl:and (cl:> dx 0) (find 'R active-bounds)))
	(check-top    (cl:and (cl:> dy 0) (find 'T active-bounds)))
	(check-bottom (cl:and (cl:< dy 0) (find 'B active-bounds)))
	(cond-left   `(cms-cuda::>= (cms-cuda::- x ,(cl:- dx)) 0))
	(cond-right  `(cms-cuda::<  (cms-cuda::+ x ,dx) w))
	(cond-top    `(cms-cuda::<  (cms-cuda::+ y ,dy) h))
	(cond-bottom `(cms-cuda::>= (cms-cuda::- y ,(cl:- dy)) 0)))
    (let ((condition))
      (cl:if check-bottom (push cond-bottom condition))
      (cl:if check-top    (push cond-top condition))
      (cl:if check-right  (push cond-right condition))
      (cl:if check-left   (push cond-left condition))
      `(border-type ,dx ,dy (,check-left ,check-right ,check-top ,check-bottom) ,condition ,@body))))


;;; loop over filter mask for current pixel and drop zeros
(defmacro loop-over-area (to-left to-right to-top to-bottom mask bounds &body body)
  (labels ((use (x y) (cl::if (cl::not (cl::or (eql (cl:nth (cl:+ x to-left)
							    (cl:nth (cl:+ y to-bottom) mask))
						    0)
					       (eql (cl:nth (cl:+ x to-left)
							    (cl:nth (cl:+ y to-bottom) mask))
						    0.0)))
			      t nil)))
    `(progn
       ,@(loop for rel-y from (cl:- to-bottom) to to-top nconc
	   (loop for rel-x from (cl:- to-left) to to-right collect
	     (cl:when (use rel-x rel-y)
		    `(let ((rx ,rel-x) (ry ,rel-y))
		       (declare (ignorable rx ry))
		       (boundary-checked-access ,rel-x ,rel-y ,bounds
			 ,@body))))))))

;;; check only needed boundaries
;;; (pairwise combination of left, right, bottom, top, and none)
(defmacro boundary-case (to-left to-right to-top to-bottom mask lr tb &body body)
  (let ((core (cl::and (eql tb '_) (eql lr '_))))
    `(progn
       ,(cl:if (cl:not core)
	       `(block-label ,(format nil "~a_~a"
				      (lisp (cond ((eq tb 'T) "top")  ((eql tb 'B) "bottom") (t "just")))
				      (lisp (cond ((eq lr 'L) "left") ((eql lr 'R) "right")  (t "only"))))))
       (loop-over-area ,to-left ,to-right ,to-top ,to-bottom ,mask ,(list lr tb) ,@body)
       (goto-block accum_done))))

;;; check every possible boundary
;;; (all: left, right, bottom, and top)
(defmacro fallback-case (to-left to-right to-top to-bottom mask &body body)
  `(progn
     (block-label fallback)
     (loop-over-area ,to-left ,to-right ,to-top ,to-bottom ,mask (L R T B) ,@body)))


;;; simd border handling  macros
;;  ==============================================

;;; cpu-block: no vectorization, no intrinsics, default accumulator initialization
(defmacro cpu-block (left right bottom top A B offsets mask input output comps init initially finally (&key fallback) &body body)
  `(progn
     ,(cl:if fallback
	     (comment "pragma omp for collapse(2)" :prefix "#")
	     (comment "pragma omp for collapse(2) nowait" :prefix "#"))
     (for ((int y = ,bottom) (< y ,top) ++y)
	 (for ((int x = ,left) (< x ,right) ++x)
	     (prepare-accums-default ,input ,comps ,init ,initially
	       ,(cl:if fallback
		       `(fallback-case ,@offsets ,mask (with-vec4 ,input ,output ,@body))
		       `(boundary-case ,@offsets ,mask ,A ,B (with-vec4 ,input ,output ,@body)))
	       (store-accums-default ,comps ,output ,finally))))))

;;; vectorize and intrisify
(defmacro simd-block (left right bottom top A B offsets mask input output comps init initially finally &body body)
  `(progn
     (comment "pragma omp for collapse(2) nowait" :prefix "#")
     (for ((int y = ,bottom) (< y ,top) ++y)
	 (for ((int x = ,left) (< x ,right) (+= x (simd-w)))
	     (simd-wrapper ,input ,output ,initially ,finally ,comps ,init
	       (boundary-case ,@offsets ,mask ,A ,B (vectorize ,input ,output (progn ,@body))))))))


;;; border handling features: sse,avx and cuda hipacc
;;; ==============================================

(define-feature border-handling ((&key (to-left 0) (to-right 0) (to-bottom 0) (to-top 0))
				 (&key mask)
				 (&key (input '()) (output '()))
				 (&key (initially nil) (finally nil) (comps 3) (init 0)) &body body))

;;; default implementation, no border processing optimization
(implement border-handling (default)
  `(loop-over-area ,to-left ,to-right ,to-top ,to-bottom ,mask (L R T B)
     (simd-wrapper ,input ,output ,initially ,finally ,comps ,init
       (block ,@body))))

;;; sse+avx loop over every border region seperately: vectorize where possible
(implement border-handling (sse)
  (let* ((offsets `(,to-left ,to-right ,to-top ,to-bottom))
	 (arg-stuff `(,offsets ,mask ,input ,output ,comps ,init ,initially ,finally)))
    `(if (and (> w (cl:max, (cl:+ 1 to-left to-right) (simd-w)))
	   (> h ,(cl:+ 1 to-top to-bottom)))
	 (progn
	   (comment "pragma omp parallel" :prefix "#")
	   (block
	     ;; CPU left bottom
	     (cpu-block 0 ,to-left 0 ,to-bottom  L B ,@arg-stuff (:fallback nil) ,@body)
	     ;; CPU left middle
	     (cpu-block 0 ,to-left ,to-bottom (- h ,(1+ to-top)) L _ ,@arg-stuff (:fallback nil) ,@body)
	     ;; CPU left top
	     (cpu-block 0 ,to-left (- h ,(1+ to-top)) h L T ,@arg-stuff (:fallback nil) ,@body)
	     ;; CPU right bottom
	     (cpu-block (- w ,to-right (simd-w)) w 0 ,to-bottom  R B ,@arg-stuff (:fallback nil) ,@body)
	     ;; CPU right middle
	     (cpu-block (- w ,to-right (simd-w)) w ,to-bottom (- h ,(1+ to-top)) R _ ,@arg-stuff (:fallback nil) ,@body)
	     ;; CPU right top
	     (cpu-block (- w ,to-right (simd-w)) w (- h ,(1+ to-top)) h R T ,@arg-stuff (:fallback nil) ,@body)
	     ;; SIMD center bottom
	     (simd-block ,to-left (- w ,to-right (simd-w)) 0 ,to-bottom B _ ,@arg-stuff ,@body)
	     ;; SIMD center Top
	     (simd-block ,to-left (- w ,to-right (simd-w)) (- h ,(1+ to-top)) h T _ ,@arg-stuff ,@body)
	     ;; SIMD center
	     (simd-block ,to-left (- w ,to-right (simd-w)) ,to-bottom (- h ,(1+ to-top)) _ _ ,@arg-stuff ,@body)
	     ))

	 ;; CPU fallback case
	 (cpu-block 0 w 0 h nil nil ,@arg-stuff (:fallback t) ,@body)
	 )))

;;; cuda hipacc border region handling: check position of kernel in image and goto specialized code
(implement border-handling (cuda hipacc)
  (let ((offsets `(,to-left ,to-right ,to-top ,to-bottom)))
    `(progn
       
       (decl ((const unsigned int first-x = (/ ,to-left blockDim.x))
	      (const unsigned int first-y = (/ ,to-bottom blockDim.y))
	      (const unsigned int last-x = (1- (- (+ (/ w blockDim.x) (? (% w blockDim.x) 1 0)) (/ ,to-right blockDim.x))))
	      (const unsigned int last-y = (1- (- (+ (/ h blockDim.y) (? (% h blockDim.y) 1 0)) (/ ,to-top blockDim.y)))))
	   (cond ((or (< w blockDim.x)
		    (< h blockDim.y)
		    (< w ,(cl::+ to-left to-right 1))
		    (< h ,(cl::+ to-bottom to-top 1))) (goto fallback))

		 ((<= blockIdx.x first-x)
		  (cond ((<= blockIdx.y first-y) (goto bottom_left))
			((>= blockIdx.y last-y) (goto top_left))
			(t (goto just_left))))
		 ((>= blockIdx.x last-x)
		  (cond ((<= blockIdx.y first-y) (goto bottom_right))
			((>= blockIdx.y last-y) (goto top_right))
			(t (goto just_right))))
		 ((<= blockIdx.y first-y) (goto bottom_only))
		 ((>= blockIdx.y last-y) (goto top_only))))
       
       (boundary-case ,@offsets ,mask _ _ (with-vec4 ,input ,output (block ,@body)))
       (boundary-case ,@offsets ,mask L T (with-vec4 ,input ,output (block ,@body)))
       (boundary-case ,@offsets ,mask _ T (with-vec4 ,input ,output (block ,@body)))
       (boundary-case ,@offsets ,mask R T (with-vec4 ,input ,output (block ,@body)))
       (boundary-case ,@offsets ,mask L _ (with-vec4 ,input ,output (block ,@body)))
       (boundary-case ,@offsets ,mask R _ (with-vec4 ,input ,output (block ,@body)))
       (boundary-case ,@offsets ,mask L B (with-vec4 ,input ,output (block ,@body)))
       (boundary-case ,@offsets ,mask _ B (with-vec4 ,input ,output (block ,@body)))
       (boundary-case ,@offsets ,mask R B (with-vec4 ,input ,output (block ,@body)))
       (fallback-case ,@offsets ,mask (with-vec4 ,input ,output (block ,@body)))
       (clabel accum_done))))



;;; codelet wrapper
;;; ==============================================

(define-feature simd-wrapper (input output initially finally comps init core))

;;; default: process image data as vec4
(implement simd-wrapper (default)
  `(with-vec4 ,input ,output ,core))

;;; sse+avx: load and process image data in registers
(implement simd-wrapper (sse)
  (let ((rgb (cl:if (cl::> comps 1) t nil)))
    `(let ((mask imask))
       (declare (ignorable mask))
       (vectorize ,input ,output
	 (decl ,(cl::if initially initially
			`(((xmm-type) xmm_r = (set-fx ,init))
			  ,@(cl:if (> comps 1)
				   `(((xmm-type) xmm_g = xmm_r)
				     ((xmm-type) xmm_b = xmm_r)))))
	     (symbol-macrolet ,(cl::if initially '()
				       '((r 'xmm_r)
					 (g 'xmm_g)
					 (b 'xmm_b)))
	       (progn ,core
		      ,(cl::if finally finally
			       `(set (,(first output) 0) xmm_r
				    ,@(cl::if rgb
					      `((,(first output) 1) xmm_g
						(,(first output) 2) xmm_b)))))))))))


;;; prepare accumulation variables
;;; ==============================================

(define-feature prepare-accums (input comps init initially &body body))

;;; default as macro since it's used in 'cpu-block' feature with sse target
(defmacro prepare-accums-default (input comps init initially &body body)
  (cl::if initially
	  `(with-vec4 ,input ()
	     (decl ,initially
		 ,@body))
	  `(decl ((float r = ,init)
		  ,@(cl:if (cl:> comps 1)
			   `((float g = ,init)
			     (float b = ,init))))
	       ,@body)))

(implement prepare-accums (default)
  `(prepare-accums-default ,input ,comps ,init ,initially ,@body))

;;; preparation for sse/avx is handled by simd-wrapper
(implement prepare-accums (sse) `(progn ,@body))


;;; prepare accumulation store operation
;;; ==============================================

(define-feature store-accums (comps output finally))

;;; default as macro since it's used in 'cpu-block' feature with sse target
(defmacro store-accums-default (comps output finally)
  `(with-vec4 nil ,output
     (progn ,(cl:if finally
		    finally
		    `(set (,(first output) 0) r
			 ,@(cl:if (cl:> comps 1)
				  `((,(first output) 1) g
				    (,(first output) 2) b)))))))

(implement store-accums (default)
  `(store-accums-default ,comps ,output ,finally))

;;; accum store for sse/avx is handled by simd-wrapper
(implement store-accums (sse) nil)


;;; prepare filter-mask for sse avx
;;; ==============================================

(define-feature prepare-mask (imask mask))


;;; introduce additional filter-mask with sse-constants
(implement prepare-mask (sse)
  (let ((fmask (loop for i in mask collect
		 (loop for k in i collect
		   `(float-type ,k)))))
    `(intrinsify :sse (hide (setf ,imask ',fmask)))))

;;; introduce additional filter-mask with avx-constants
(implement prepare-mask (avx)
  (let ((fmask (loop for i in mask collect
		 (loop for k in i collect
		   `(float-type ,k)))))
    `(intrinsify :avx (hide (setf ,imask ',fmask)))))


;;; point core operation
;;; ==============================================

(define-feature point-core ((&key input output) &body body))

;;; loop over linear image array
(implement point-core (default)
  `(progn
     (comment "pragma omp prallel for" :prefix "#")
     (for ((int i = 0) (< i (* w h)) ++i)
	 (with-vec4 ,input ,output ,@body))))

;;; no loop implementation, use vec4
(implement point-core (cuda)
  `(with-vec4 ,input ,output ,@body))

;;; SIMD processing linear image array
;;; scalar processing of remaining array alements
(implement point-core (sse)	   
  `(decl ((const int picSize = (* w h))
	  (const int arraySize = (- picSize (% picSize (simd-w)))))
       (comment "pragma omp parallel" :prefix "#")
       (block
	 (vectorize ,input ,output
	   (progn
	     (comment "pragma omp for nowait" :prefix "#")
	     (for ((int i = 0) (cms-cuda::< i arraySize) (cms-cuda::+= i (simd-w))) ,@body)))
	 (comment "pragma omp for nowait" :prefix "#")
	 (for ((int i = arraySize) (< i picSize) ++i)
	     (with-vec4 ,input ,output ,@body)))))


;;; ==============================================
;;;               operator types
;;; ==============================================


;; debug
(defmacro defdebug (name (&key input output arch) &body body)
  `(with-config (make-config ,@(ensure-list arch) point-operator)
     (function-setup ,name (:input ,input :output ,output)
		     (point-core (:input ,input :output ,output) ,@body))))


;;; point operator
(defmacro defpoint (name (&key input output arch) &body body)
  `(with-config (make-config ,@(ensure-list arch) point-operator) 
     (macrolet ,(loop for array in (append input output) collect `(,array () `(abs-array ,',array)))    
       (function-setup ,name (:input ,input :output ,output)
	 (point-core (:input ,input :output ,output) ,@body)))))


;;; local operator
(defmacro deflocal (name (&key input output arch) &key mask (operator '*) (aggregator '+) (grayscale nil) (init-val 0.0f) (codelet nil) (finally nil) (initially nil))
  (let* ((init init-val)
	 (comps (cl:if grayscale 1 3))
	 (mask-w (cl:length mask))
	 (mask-h (cl:length (first mask)))
	 (mask-extend-x (lisp (floor (/ mask-w 2))))
	 (mask-extend-y (lisp (floor (/ mask-h 2))))
	 (inputs input)
	 (outputs output)
	 (input (first input))
	 (output (first output)))
    (declare (ignorable output))
    (format t "// -----> local operator ~a  cmp: ~a, iv: ~a~%" name comps init)    
    `(with-config (make-config ,@(ensure-list arch))
       (macrolet (,@(loop for array in inputs collect `(,array (rx ry) `(rel-array ,',array ,rx ,ry)))
		  ,@(loop for array in outputs collect `(,array () `(abs-array ,',array))))
	 (function-setup ,name (:input ,inputs :output ,outputs)
	   (let ((mask ',mask)
		 (imask nil))
	     (declare (ignorable mask imask))
	     (progn
	       (prepare-mask imask ,mask) 
	       (loop-setup 
		(prepare-accums ,inputs ,comps ,init ,initially				
		  (border-handling (:to-left ,mask-extend-x :to-right ,mask-extend-x
				    :to-bottom ,mask-extend-y :to-top ,mask-extend-y)
				   (:mask ,mask)
				   (:input ,inputs :output ,outputs)
				   (:initially ,initially :finally ,finally :comps ,comps :init ,init)			       
		     ,(cl::if (cl::not codelet)
			      `(progn
			      	 ,@(let ((vars '(r g b)))
			      	     (loop for c from 0 to (cl:1- comps) collect
			      	       `(set ,(nth c vars) (,aggregator ,(nth c vars)
			      						(,operator (float-type (cl::nth (cl::+ rx ,mask-extend-x)
			      										(cl::nth (cl::+ ry ,mask-extend-y) mask)))
			      							   (,input rx ry ,c)))))))
			      codelet))
		  (store-accums ,comps ,outputs ,finally)
		  )))))))))


;;; ==============================================
;;;                   image
;;; ==============================================

(defun arch-storage (arch)
  (cl::if (listp arch)
	  (cl::if (cl::or (find 'cuda arch)
		    (find :cuda arch))
		  'cuda
		  'cpu)
	  (cl::if (cl::or (eql arch 'cuda)
		    (eql arch :cuda))
		  'cuda
		  'cpu)))

(defmacro load-image (name (&key input output arch) &key file)
  (declare (ignore input arch))
  (lisp (if (not (= (cl:length output) 1)) (error "Cannot load multiple images in one load-image call [~a]" output)))
  (let ((out (first output)))
    `(function ,name ((ub_image& ,out)) -> void
	 (set ',out (funcall load-ub-image ,file)))))

(defmacro store-image (name (&key input output arch) &key file)
  (declare (ignore output arch))
  (lisp (if (not (= (cl:length input) 1)) (error "Cannot store multiple images in one store-image call [~a]" input)))
  (let ((save (first input)))
    `(function ,name ((ub_image& ,save)) -> void
	 (funcall store-ub-image ,save ,file))))

(defmacro transition (name (&key input output arch) &key from to)
  (declare (ignore arch))
  (cl:if (eql to from) (error "Bad transition from ~a to ~a." from to))
  `(function ,name ,(cl:if (eql (arch-storage from) 'cpu)
			   (append (loop for i in input collect `(ub_image& ,i))
				   (loop for i in output collect `(ub_image_cuda& ,i)))
			   (append (loop for i in input collect `(ub_image_cuda& ,i))
				   (loop for i in output collect `(ub_image& ,i))))
       -> void
       (checked-cuda ,(format nil "transition ~a -> ~a (~a)" from to name)
	 ,@(loop for (src dst) in (mapcar (lambda (a b) (list a b)) input output)
		 collect (cl:if (eql (arch-storage from) 'cpu)
				`(upload-to-cuda ',src ',dst)
				`(download-from-cuda ',src ',dst))))))

(defmacro allocator (name (&key input output arch))
  (declare (ignore input output arch))
  `(function ,name () -> void
       (checked-cuda "allocator"
	 (funcall allocate-images))))


;(use-functions upload-to-cuda download-from-cuda)


;;; ==============================================
;;;                   graph
;;; ==============================================

(lisp (defun topo-sort (graph)
	(let ((copy-nodes (loop for i in graph collect `(,(slot-value i 'name)
							 ,(slot-value i 'predecessors)
							 ,(slot-value i 'successors))))
	      (targ-nodes nil)
	      (test (make-hash-table :test 'equal)))
	  (loop while (not (eql (cl:length copy-nodes) 0)) do
	    (let ((remove-this-transition (make-hash-table))
		  (remove-this-image  (make-hash-table))
		  (remove-not-yet (make-hash-table))
		  (next-iter nil))
	      (mapcar #'(lambda (x) (if (eql (second x) nil)
					(progn
					  (push (first x) targ-nodes)
					  (setf (gethash (first x) remove-this-transition) t)
					  (mapcar #'(lambda (y) (setf (gethash y remove-this-image) t)) (third x)))
					(progn
					  (mapcar #'(lambda (y) (setf (gethash y remove-not-yet) t)) (third x))))) copy-nodes)
	      (mapcar #'(lambda (x) (if (not (gethash (first x) remove-this-transition))
					(push `(,(first x) ,(remove-if #'(lambda (y) (and (gethash y remove-this-image)
										       (not (gethash y remove-not-yet))))
								       (second x)) ,(third x)) next-iter))) copy-nodes)
	      (setf copy-nodes next-iter)
	      
	      (if (gethash copy-nodes test)
		  (progn
		    (error "graph inconsistent")
		    (return '()))
		  (setf (gethash copy-nodes test) t))
	      ;;(format t "~s~%" next-iter)
	      ))
	  (let ((sorted nil))
	    (loop for i in targ-nodes do
	      (loop for k in graph do
		(if (eql i (slot-value k 'name))
		    (push k sorted))))
	    (format t "/*---- sorted edges: ~%~{ - ~a~%~}*/~%" sorted)
	    sorted))))

(defclass edge ()
  ((name :initarg :name :accessor name)
   (operator :initarg :op :accessor operator)
   (predecessors :initarg :pred :accessor predecessors)
   (successors :initarg :succ :accessor successors)
   (transition :initform nil :initarg :transition :accessor storage-transition)
   (arch :initform 'cpu :initarg :arch :accessor arch)
   (scale-factor-x :initform 1 :accessor scale-factor-x)
   (scale-factor-y :initform 1 :accessor scale-factor-y)))

(defclass graph-node ()
  ((name :initarg :name :accessor name)
   (stored-on :initform :unclear :initarg :stored-on :accessor stored-on)
   (requires-allocation :initform t :accessor requires-allocation)
   (has-mipmaps :initform nil :accessor has-mipmaps)))

(defmethod print-object ((edge edge) stream)
  (format stream "#<graph-edge ~a>" (name edge)))
(defmethod print-object ((node graph-node) stream)
  (format stream "#<graph-node ~a>" (name node)))


(defmethod show ((e edge))
  (format t "/* EDGE ~a~%      pre:  ~a~%      post: ~a */~%" (name e) (predecessors e) (successors e)))

(defun draw-graph (file edges nodes)
  (with-open-file (out file :direction :output :if-exists :supersede)
    (flet ((fix-name (entity) (map 'string (lambda (x)
					     (cl:if (char-equal #\- x) #\_ x))
				   (format nil "~a" entity))))
      (format out "digraph x {~%")
      (loop for n in nodes do (format out "    ~a [shape=box,label=\"~a\\n[storage: ~a, ~a]\"];~%" (fix-name (name n)) (name n) (arch-storage (stored-on n)) (requires-allocation n)))
      (loop for e in edges do (format out "    ~a [label=\"~a\\n[arch: ~a]\"];~%~{    ~a -> ~a;~%~}~%"
				      (fix-name (name e)) (name e) (arch e)
				      (mapcar #'fix-name
					      (append (loop for suc in (successors e) append (list (name e) (cl:if (symbolp suc) suc (name suc))))
						      (loop for pre in (predecessors e) append (list (cl:if (symbolp pre) pre (name pre)) (name e)))))))
      (format out "}~%"))))

(lisp
  (defun find-nodes (edges)
    (let* ((images (remove-duplicates
		    (loop for e in edges append (append (predecessors e) (successors e)))
		    :from-end t))
	   (nodes (loop for i in images collect (make-instance 'graph-node :name i))))
      (flet ((find-node (look-for)
	       (loop for n in nodes if (equal (name n) look-for) return n
		     finally (error "cannot find node ~a [nodes: ~a]" look-for nodes))))
	(dolist (e edges)
	  (let ((new-pred (mapcar #'find-node (predecessors e)))
		(new-succ (mapcar #'find-node (successors e))))
	    (setf (predecessors e) new-pred)
	    (setf (successors e) new-succ)))
	nodes)))

  (defun add-transitions (edges nodes)
    (let ((new-edges))
      (labels ((fix-transition (e p)
		 ;; we start in this configuration:
		 ;;    {pre} --> [img p] --> [edge e] --> {suc}
		 (let* ((target-arch (arch e))
			(source-arch (stored-on p))
			;; and build [img T]
			(transition-image-name (cintern (symbol-name (gensym "transition-img"))))
			(transition-image (make-instance 'graph-node
							 :name transition-image-name
							 :stored-on target-arch))
			;; and [edge T]
			(transition-edge-name (cintern (symbol-name (gensym "transfer"))))
			(transition-edge (make-instance 'edge
							:name transition-edge-name
							:transition target-arch
							:pred (list p)
							:succ (list transition-image)
							:arch :memcpy
							:op `(transition ,transition-edge-name
								 (:input ,(list (name p)) :output ,(list transition-image-name))
								 :from ,source-arch :to ,target-arch))))
		   ;; now we have
		   ;;     {pre} --> [img p] --> [edge T] --> [img T] --> [edge e] --> {suc}
		   ;; so make sure all {suc} know where they should be allocated
		   (dolist (s (successors e))
		     (setf (stored-on s) target-arch))
		   ;; and add the nodes/edges
		   (push transition-edge new-edges)
		   (push transition-image nodes)
		   transition-image))
	       ;; if we find a matching [img]->[edge] configuration we still have to propagate the storage information
	       (propagate-arch (e arch)
		 (dolist (s (successors e))
		   (setf (stored-on s) arch))))
	;; go over all edges and insert transition nodes where appropriate
	(dolist (e edges)
	  (setf (predecessors e)
		(loop for p in (predecessors e) collect
						(if (equal (arch-storage (stored-on p)) (arch-storage (arch e)))
						    (progn (propagate-arch e (stored-on p))
							   p)
						    (fix-transition e p))))
	  (push e new-edges))
	(values (reverse new-edges) nodes))))
  
  (defun execution-plan (edges nodes) 
    "Returns the edges `sorted', currently that means in file order.
    However, we collect all `input-edges' and put them first in the list."
    (declare (ignore nodes))
    (multiple-value-bind (seeds depended)
	(loop for x in (topo-sort edges)
	      if (cl:null (predecessors x)) collect x into s 
		else collect x into d finally (return (values s d)))
      (dolist (s seeds)
	(setf (arch s) 'cpu)
	(dolist (i (successors s))
	  (setf (stored-on i) 'cpu)))
      (let ((alloc (make-instance 'edge :name 'image-allocator :pred nil :succ nil
					:op `(allocator image-allocator ()))))
	(append seeds (list alloc) depended))))
  
  (defun track-dependencies (edges nodes)
    (declare (ignore nodes))
    ;; find transitions - do we still need this?
    (loop for e in edges do 
      (if (equal (first (operator e)) 'transition) 
	  (let* ((flat (flatten (operator e)))
		 (to (cadr (member :to flat))))
	    (setf (storage-transition e) to))))
    ;; determine storage space for nodes
    ;; we assume the edges are already in proper execution order
    (dolist (e edges)
      (if (cl:null (predecessors e))
	  ;; edges with no predecessors (terminology??) are load-edges, and also need not be allocated
	  (loop for s in (successors e) do 
	    (progn (setf (stored-on s) 'cpu)
		   (setf (requires-allocation s) nil)))
	  ;; otherwise check if all predecessors argree
	  (let ((pred-storage (remove-duplicates (loop for pred in (predecessors e) collect (stored-on pred)))))
	    ;; can an edge have multiple preds?? this is also strange...
	    (if (not (= (cl:length pred-storage) 1)) (error "Not all predecessors of ~a agree on storage: ~a" (name e) pred-storage)) 
	    (let ((location (if (storage-transition e) (storage-transition e) (first pred-storage))))
	      (dolist (s (successors e))
		(setf (stored-on s) location)))))))
  
  (defun propagate-image-scale (plan)
    "The PLAN has to be topologically sorted for this to work.
    Also, I'm not sure how to manage different input sizes, we'll ignore that for now."
    `(progn
       ,@(loop for e in plan collect
			     (let ((base (first (predecessors e))))
			       (loop for img in (successors e)
				     if (requires-allocation img)
				       append `(set (oref ,(name img) w) (* ,(scale-factor-x e) (oref ,(name base) w))
						   (oref ,(name img) h) (* ,(scale-factor-y e) (oref ,(name base) h)))
				     if (not (requires-allocation img))
				       append `(comment ,(format nil "image ~a does not require allocation." (name img)))
				     )))))
  
  
  
  )

(defmacro filter-graph (name &body body)
  (let ((edges (eval `(let (edges)
			(macrolet ((edge (name (&key input output arch) operator)
				     `(let ((inputs (ensure-list ',input))
					    (outputs (ensure-list ',output)))
					(push (make-instance 'edge :name ',name
								   :pred inputs
								   :succ outputs
								   :arch ',(cl:if arch arch 'cpu)
								   :op (destructuring-bind (op &body body) ',operator
									 (let ((arch-spec ',(cl:if arch arch 'cpu)))
									   `(,op ,',name (:input ,inputs :output ,outputs :arch ,arch-spec) ,@body))))
					      edges))))
			  ,@body)
			edges))))
    (let* ((nodes (find-nodes edges))
	   (_ (draw-graph "graph0-init.dot" edges nodes))
	   (plan (execution-plan edges nodes)))
      (declare (ignore _))
      (draw-graph "graph1-plan.dot" plan nodes)
      (multiple-value-bind (plan nodes) (add-transitions plan nodes)
	(track-dependencies plan nodes)
	(draw-graph "graph2-deps.dot" plan nodes)
	(flet ((image-type (img)
		 (case (arch-storage (stored-on img))
		   (cuda 'ub_image_cuda)
		   (cpu 'ub_image)
		   (otherwise 'unknown_type))))
	  `(progn 
	     (comment ,(format nil "This is the filter graph ~a." name))
	     ;; first declare all images
	     (decl ,(loop for img in nodes collect 
			  `(,(image-type img) ,(name img)))
		 ;; and provide an allocator
		 (function allocate-images () -> void
		     ;; propagate... takes care that each image's dimensions are written to img.[xy]
		     ,(propagate-image-scale plan)
		     ,@(loop for img in nodes if (requires-allocation img) collect
			     `(progn
				(cout "allocating image " ,(format nil "~a" (name img)) " with size " (oref ,(name img) w) " x " (oref ,(name img) h) endl)
				(set ,(name img) (funcall ,(image-type img) (oref ,(name img) w) (oref ,(name img) h))))))
		 ;; now instantiate all filters
		 ;;,(loop for e in plan do (format t "/*operator ~a: ~a*/~%" (name e) (macroexpand-1 (operator e))))
		 ,@(loop for e in plan collect (operator e))
		 ;; now build main function
		 (function main () -> int
		     ,@(loop for e in plan collect
			     `(funcall ,(name e)
				  ,@(loop for i in (predecessors e) append (list (name i)))
				  ,@(loop for i in (successors e) append (list (name i)))))))))))))


;(use-variables cudaSuccess)

(defmacro checked-cuda (text &body body)
  `(progn
     (error-on-bad-cuda-result (cudaGetLastError) ,(format nil "precondition to ~a" text))
     ,@body
     (error-on-bad-cuda-result (cudaGetLastError) ,text)
     (error-on-bad-cuda-result (cudaDeviceSynchronize) ,(format nil "sync after ~a" text))))

(defmacro chp-preamble ()
  `(progn (include <iostream>)
	  (include <string>)
	  (include <cuda.h>)
	  (include <stdio.h>)
	  (include <immintrin.h>)
	  (include <emmintrin.h>)
	  (include "avx_mathfun.h")
	  (include "sse_mathfun.h")
	  (include "image.h")
	  (using-namespace std)

	  (function error-on-bad-cuda-result ((cudaError_t code) (const char *text)) -> void
	      (when (!= code cudaSuccess)
		(cerr "CUDA ERROR on " text ":" endl)
		(cerr "CUDA ERROR: " (cudaGetErrorString code) endl)
		(exit -1)))))
