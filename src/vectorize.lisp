(in-package :chipotle)

;;; ==============================================
;;;              SSE vectorization
;;; ==============================================

(defparameter *iftrinsify* '(()))

(lisp
  (defun find-var (var)
    (loop for i in (first *iftrinsify*) do
      (loop for k in i do
	(if (equal var k)
	    (return t))))))


;;; load data from array to simd register
(define-feature load-aligned (value))

;;; unaligned load as default
(implement load-aligned (default)        `(xmm-loadu (cast const (postfix* (xmm-type i)) (addr-of ,value))))

; TODO fox: missing aligned array
;;; aligned load for point operators
;(implement load-aligned (point-operator) `(xmm-load  (cast const (postfix* (xmm-type i)) (addr-of ,value)))) 

;;; store data from simd register to array
(define-feature store-aligned (value var))

;;; unaligned store as default
(implement store-aligned (default)        `(xmm-storeu (cast (postfix* (xmm-type i)) (addr-of ,value)) ,var))

; TODO fix: missing aligned array
;;; aligned store for point operators
;(implement store-aligned (point-operator) `(xmm-store  (cast (postfix* (xmm-type i)) (addr-of ,value)) ,var))

;;; prepare temporary variables for load operations
(define-feature load-intrin-head (variable value &body body))

;;; unsigned char specific 
(implement load-intrin-head (unsigned-char)
  ;; if variable is already prepared skip
  (cl:if (cl:not (gethash variable prep-done))
	 (cl:progn
	   ;; set info that variable is prepared
	   (setf (gethash variable prep-done) t)
	   `(progn
	      (comment (format nil "Load: ~a to ~a" ',value ',(first variable)))
	      ;; load data from array to simd register
	      (cms-cuda::decl ((const (xmm-type i) ,(first variable) = (load-aligned ,value)))
		  ,@body)))
	 `(progn ,@body)))

;;; floating point specific / TODO not finished yet
(implement load-intrin-head (floating-point)
  (cl:if (cl:not (gethash variable prep-done))
	 (let ((var1 (intrinsic-tmp "rgba_"))
	       (var2 (intrinsic-tmp "rgba_"))
	       (var3 (intrinsic-tmp "rgba_"))
	       (var4 (intrinsic-tmp "rgba_"))
	       (var5 (intrinsic-tmp "rg_"))
	       (var6 (intrinsic-tmp "rg_"))
	       (var7 (intrinsic-tmp "ba_"))
	       (var8 (intrinsic-tmp "ba_")))
	   (setf (gethash variable prep-done) t)
	   `(progn
	      (comment (format nil "Load: ~a to ~a" ',value ',variable))
	      (cms-cuda::decl ((const __m128 ,var1 = (_mm_load_ps (cast const float* (addr-of ,value))))
			      (const __m128 ,var2 = (_mm_load_ps (cms-cuda::+ (cast const float* (addr-of ,value)) 4)))
			      (const __m128 ,var3 = (_mm_load_ps (cms-cuda::+ (cast const float* (addr-of ,value)) 8)))
			      (const __m128 ,var4 = (_mm_load_ps (cms-cuda::+ (cast const float* (addr-of ,value)) 12)))
			      (const __m128 ,var5 = (_mm_unpacklo_ps ,var1 ,var2))
			      (const __m128 ,var6 = (_mm_unpacklo_ps ,var3 ,var4))
			      (const __m128 ,var7 = (_mm_unpackhi_ps ,var1 ,var2))
			      (const __m128 ,var8 = (_mm_unpackhi_ps ,var3 ,var4))
			      (const __m128 ,(first variable) = (_mm_unpacklo_ps ,var5 ,var6))
			      (const __m128 ,(second variable) = (_mm_unpackhi_ps ,var5 ,var6))
			      (const __m128 ,(third variable) = (_mm_unpacklo_ps ,var7 ,var8))
			      (const __m128 ,(fourth variable) = (_mm_unpackhi_ps ,var7 ,var8)))
		  ,@body)))
	 `(progn ,@body)))


;;; prepare temporary variables for store operations
(defmacro store-intrin-head (variable value &body body)
  ;; check and update store attempts.
  ;; used to identify the final store operation.
  (let ((store-attempts (gethash variable store-count)))
    (cl::if store-attempts
	    (setf (gethash variable store-count) (cl:+ 1 store-attempts))
	    (setf (gethash variable store-count) 1)))

  ;; pack variables and write to array
  (let* ((store-tmp (gensym "store-count-"))
	 (var1 (intrinsic-tmp "r_"))
	 (var2 (intrinsic-tmp "g_"))
	 (var3 (intrinsic-tmp "b_"))
	 (var4 (intrinsic-tmp "a_"))
	 (var5 (intrinsic-tmp "rg_"))
	 (var6 (intrinsic-tmp "ba_"))
	 (var7 (intrinsic-tmp "rgba_"))
	 (store `(let ((,store-tmp (cl:- (gethash ',variable store-count) 1)))
		   (setf (gethash ',variable store-count) ,store-tmp)
		   ;; the for final store operation
		   (cl:when (cl:eq ,store-tmp 0)
		     (progn
		       (comment (format nil "Store: ~a to ~a" ',variable ',value))
		       (cms-cuda::decl
			   ;; convert float to uchar and shift to proper byte position
			   ((const (xmm-type i) ,var1 =  (cvt-f2i      ,(first  variable)))
			    (const (xmm-type i) ,var2 = (<<< (cvt-f2i ,(second variable)) 1))
			    (const (xmm-type i) ,var3 = (<<< (cvt-f2i ,(third  variable)) 2))
			    (const (xmm-type i) ,var4 = (<<< (cvt-f2i ,(fourth variable)) 3))
			    ;; recombine to single uchar4 with rgba alignment
			    (const (xmm-type i) ,var5 = (or2 ,var1 ,var2))
			    (const (xmm-type i) ,var6 = (or2 ,var3 ,var4))
			    (const (xmm-type i) ,var7 = (or2 ,var5 ,var6)))
			   ;; write to array
			   (store-aligned ,value ,var7)))))))

    ;; check if preparations for store variables are already done
    (cl::if (cl:not (gethash variable prep-done))
  	    ;; prepare temporary variables for store operations
  	    (cl:progn
  	      (setf (gethash variable prep-done) t)
  	      `(progn
  		 (comment (format nil "Prepare store variable: ~a" ',variable))
  		 (cms-cuda::decl (((xmm-type) ,(first variable) = (set-f0))
  				 ((xmm-type) ,(second variable) = (set-f0))
  				 ((xmm-type) ,(third variable) = (set-f0))
  				 ((xmm-type) ,(fourth variable) = (set-f0)))
				 ;; let evtl weg machen wenn let in decl 
		      (let ((,(first variable) ',(first variable))
			    (,(second variable) ',(second variable))
			    (,(third variable) ',(third variable))
			    (,(fourth variable) ',(fourth variable)))
			(progn
			  ,@body
			  ;; if it's not the final store attempt 'store' is empty
			  ,store)))))
  	    `(let ((,(first variable) ',(first variable))
  		   (,(second variable) ',(second variable))
  		   (,(third variable) ',(third variable))
  		   (,(fourth variable) ',(fourth variable)))
  	       (progn
  		 ,@body
  		 ;; it it's not the final store attempt 'store' is empty
  		 ,store)))))

;;; load single color (r,g,b,a) to register
(define-feature register-load (value col))

;;; load multiple uchar4 (rgba) and seperate specific channel to simd register
(defmacro register-load+shift (value left right)
  ;; convert uchar to float
  `(cvt-i2f
    ,(cl:cond
       ;; shift in zeros, get rid of other colors
       ((cl:eql left 0) `(>> ',value (cl:* 8 ,right)));;TODO FIX value quote!!!!!! somewhere something's wrong 
       ((cl:eql right 0) `(<< ',value (cl:* 8 ,left)))
       ((cl:eql left 3) `(>>> (<< ',value (cl:* 8 ,left)) ,right))
       (t `(>> (<< ',value (cl:* 8 ,left)) (cl:* 8 ,right))))))


;; load uchar4 coded colors
(implement register-load (unsigned-char)
  (cl:cond
    ((cl:eql col 'red)   `(register-load+shift ,(first value) 3 3))
    ((cl:eql col 'green) `(register-load+shift ,(first value) 2 3))
    ((cl:eql col 'blue)  `(register-load+shift ,(first value) 1 3))
    ((cl:eql col 'alpha) `(register-load+shift ,(first value) 0 3))))

;; load float colors
(implement register-load (floating-point)
  (cl:cond
    ((cl:eql col 'red)   (first value))
    ((cl:eql col 'green) (second value))
    ((cl:eql col 'blue)  (third value))
    ((cl:eql col 'alpha) (fourth value))))

(defmacro vectorize# (instruction-set input output &body body)
  (let ((hashkey (gensym))
	(hashkey2 (gensym)))
    (setf (gethash hashkey preparations) (make-hash-table :test 'equal))
    (setf (gethash hashkey2 preparations) (make-hash-table :test 'equal))
    
    `(intrinsify ,instruction-set
       (let ((if-cond nil)
	     (intrinsified t))
	 (declare (ignorable if-cond intrinsified))

	 (macrolet (,@(loop for i in (append input output) collect
			`(,i (&rest rest)
			     (cl:if (cl:eq (cl:length rest) 0)
				    `(abs-array ,',i ,@rest)
				    `(rel-array ,',i ,@rest))))
		    
		    (set (variable value &rest rest)
			;; TODO fuse with cuda code
			(let ((prepare-load (gethash ',hashkey preparations))
			      (prepare-store (gethash ',hashkey2 preparations))
			      (prepared nil)
			      (xmm1 (intrinsic-tmp)))

			  (labels
			      ((find-inputs-outputs (list)
				 (cl:when (listp list)
				   (loop for i in list do
				     (cl:when (listp i)
				       (cl:cond
					 ((find (car i) ',input)
					  (cl:when (cl:not (gethash (butlast i) prepare-load))
					    (setf (gethash (butlast i) prepare-load)
						  `(,(intrinsic-tmp) ,(intrinsic-tmp)
						    ,(intrinsic-tmp) ,(intrinsic-tmp)))))

					 
					 ((find (car i) ',output)
					  (cl:when (cl:not (gethash (butlast i) prepare-store))
					    (setf (gethash (butlast i) prepare-store)
						  `(,(intrinsic-tmp) ,(intrinsic-tmp)
						    ,(intrinsic-tmp) ,(intrinsic-tmp)))))
					 
					 (t (find-inputs-outputs i))))))))

			    (find-inputs-outputs `(,variable ,value))

			    (setf prepared `(progn
					      (cl:if (cl:or (cl:null if-cond) (find-var ',variable))
						     (cms-cuda::set ,variable ,value)
						     (cms-cuda::decl ((const (xmm-type) ,xmm1 = (and if-cond ,value)))
							 (cms-cuda::set ,variable (or ,xmm1 (andnot if-cond ,variable)))))
					      ,(cl:if rest `(set ,@rest))))
			    

			    (setf prepared
				  `(macrolet 
				       ,(append (loop for i in ',input collect
						  `(,i (&rest rest)
						       (declare (ignorable rest))
						       (cl:cond
							 ,@(loop for k being the hash-keys of prepare-load append
							     (cl:when (equal i (first k))
							       `(((equal rest ',(append (rest k) '(0))) '(register-load ,(gethash k prepare-load) red))
								 ((equal rest ',(append (rest k) '(1))) '(register-load ,(gethash k prepare-load) green))
								 ((equal rest ',(append (rest k) '(2))) '(register-load ,(gethash k prepare-load) blue))
								 ((equal rest ',(append (rest k) '(3))) '(register-load ,(gethash k prepare-load) alpha))))))))
						(loop for i in ',output collect
						  `(,i (&rest rest)
						       (declare (ignorable rest))
						       (cl::cond
							 ,@(loop for k being the hash-keys of prepare-store append
							     (let ((vars (gethash k prepare-store)))
							       (cl::when (equal i (first k))
								 `(((equal rest '(0)) ',(car vars))
								   ((equal rest '(1)) ',(second vars))
								   ((equal rest '(2)) ',(third vars))
								   ((equal rest '(3)) ',(fourth vars))))))))))
				     ,prepared))

			    (maphash #'(lambda (key value)
					 (setf prepared `(store-intrin-head ,value ,key ,prepared))) prepare-store)

			    (maphash #'(lambda (key value)
					 (setf prepared `(load-intrin-head ,value ,key ,prepared))) prepare-load)

			    prepared)))
		    
		    (decl (declarations &body body)
			(let ((prepare-load (gethash ',hashkey preparations))
			      (prepared nil)
			      (var-names (remove nil (loop for i in declarations collect
						       (cl::if (eql (first (last (butlast i))) '=)
							       (first (last (butlast i 2)))
							       (first (last i)))))))
						       ;(cgen::get-declaration-name i)))))
			  
			  (labels
			      ;;TODO fuse with similar functions
			      ((find-inputs (list)
				 (cl:when (listp list)
				   (loop for i in list do
				     (cl:when (listp i)
				       (cl:cond
					 ((find (car i) ',input)
					  (cl:when (cl:not (gethash (butlast i) prepare-load))
					    (setf (gethash (butlast i) prepare-load)
						  `(,(intrinsic-tmp) ,(intrinsic-tmp)
						    ,(intrinsic-tmp) ,(intrinsic-tmp)))))					 
					 (t (find-inputs i))))))))
			    
			    (find-inputs declarations)

			    (setf prepared
				  `(cms-cuda::decl ,(loop for i in declarations collect
						     (loop for k in i collect (cl:if (equal k 'float) (xmm-type)  k)))
				       (progn
					 (hide (push ',var-names (first *iftrinsify*)))
					 ,@body
					 (hide (pop (first *iftrinsify*))))))
			    
			    (setf prepared
				  `(macrolet ,(loop for i in ',input collect
						`(,i (&rest rest)
						     (declare (ignorable rest))
						     (cl:cond
						       ,@(loop for k being the hash-keys of prepare-load append
							   (cl:when (equal i (first k))
							     `(((equal rest ',(append (rest k) '(0))) '(register-load ,(gethash k prepare-load) red))
							       ((equal rest ',(append (rest k) '(1))) '(register-load ,(gethash k prepare-load) green))
							       ((equal rest ',(append (rest k) '(2))) '(register-load ,(gethash k prepare-load) blue))
							       ((equal rest ',(append (rest k) '(3))) '(register-load ,(gethash k prepare-load) alpha))))))))
				     ,prepared))

			    (maphash #'(lambda (key value)
					 (setf prepared `(load-intrin-head ,value ,key ,prepared))) prepare-load)

			    prepared)))

		    (if (test if-body &optional else-body)
			(let ((condition (intrinsic-tmp 'cond_))
			      (true-mask (intrinsic-tmp 'mask_))
			      (false-mask (intrinsic-tmp 'mask_)))

			  `(decl ((const (xmm-type) ,condition = ,test))
			       (progn
				 (hide (push '() *iftrinsify*))
				 ,(cl:if if-body
					 `(decl ((const (xmm-type) ,true-mask = (cl:if (cl::null if-cond)
										     ,condition
										     (and if-cond ,condition))))
					      (let ((if-cond ',true-mask))
						(declare (ignorable if-cond))
						,if-body)))
				 ,(cl:if else-body
					 `(decl ((const (xmm-type) ,false-mask = (cl:if (cl:null if-cond)
										      (andnot ,condition (== ,condition ,condition))
										      (andnot ,condition if-cond))))
					      (let ((if-cond ',false-mask))
						(declare (ignorable if-cond))
						,else-body)))
				 (hide (pop *iftrinsify*))))))
		    )
	   ,@body)))))


;;; ==============================================
;;;             Cuda vectorization
;;; ==============================================


(define-feature load-vec4-head (variable value &body body))

(implement load-vec4-head (unsigned-char)
  (cl:if (cl:not (gethash variable prep-done))
	 (cl:progn ;() ;(uc (intrinsic-tmp "uc_")))
	   (setf (gethash variable prep-done) t)
	   `(progn
	      (comment (format nil "Load: ~a to ~a" ',value ',variable))
	      (cms-cuda::decl ((const uchar4& ,variable = ,value))
		  ,@body)))
	 `(progn ,@body)))

(implement load-vec4-head (floating-point)
  (cl:if (cl:not (gethash variable prep-done))
	 (cl:progn
	   (setf (gethash variable prep-done) t)
	   `(progn
	      (comment (format nil "Load: ~a to ~a" ',value ',variable))
	      (cms-cuda::decl ((const float4& ,variable = ,value))	     
		  ,@body)))
	 `(progn ,@body)))

(define-feature store-vec4-head (variable array &body body))

(implement store-vec4-head (unsigned-char)
  ;;Todo check why 'progn' is required here / works without 'progn' in plain macro
  (cl:progn
    (let ((store-attempts (gethash variable store-count)))
      (cl::if store-attempts
	      (setf (gethash variable store-count) (cl:+ 1 store-attempts))
	      (setf (gethash variable store-count) 1)))
    
    (let* ((store-tmp (gensym "store-count-"))
	   (store `(let ((,store-tmp (cl:- (gethash ',variable store-count) 1)))
		     (setf (gethash ',variable store-count) ,store-tmp)
		     (cl:when (cl:eq ,store-tmp 0)
		       (progn
			 (comment (format nil "Store: ~a to ~a" ',variable ',array))
			 (cms-cuda::set ,array (funcall make-uchar4
						  (cast unsigned char (oref ,variable x))
						  (cast unsigned char (oref ,variable y))
						  (cast unsigned char (oref ,variable z))
						  (cast unsigned char (oref ,variable w)))))))))

      (cl:if (cl:not (gethash variable prep-done))
	     (cl:progn
	       (setf (gethash variable prep-done) t)
	       `(progn
		  (comment (format nil "Prepare store variable: ~a" ',variable))
		  (cms-cuda::decl ((float4 ,variable = (funcall make-float4 0.0f 0.0f 0.0f 0.0f)))
		      ,@body
		      ,store)))
	     `(let ((,variable ',variable))
		(progn
		  ,@body
		  ,store))))))

(implement store-vec4-head (floating-point)
  (cl:progn
    (let ((store-attempts (gethash variable store-count)))
      (cl::if store-attempts
	      (setf (gethash variable store-count) (cl:+ 1 store-attempts))
	      (setf (gethash variable store-count) 1)))

    (let* ((store-tmp (gensym "store-count-"))
	   (store `(let ((,store-tmp (cl:- (gethash ',variable store-count) 1)))
		     (setf (gethash ',variable store-count) ,store-tmp)
		     (cl:when (cl:eq ,store-tmp 0)
		       (progn
			 (comment (format nil "Store: ~a to ~a" ',variable ',array))
			 (cms-cuda::set ,array ,variable))))))

      (cl:if (cl:not (gethash variable prep-done))
	     (cl:progn
	       (setf (gethash variable prep-done) t)
	       `(progn
		  (comment (format nil "Prepare store variable: ~a" ',variable))
		  (cms-cuda::decl ((float4 ,variable = (funcall make-float4 0.0f 0.0f 0.0f 0.0f)))
		      ,@body
		      ,store)))
	     `(let ((,variable ',variable))
		(progn
		  ,@body
		  ,store))))))

(defmacro with-vec4 (input output &body body)
  (let ((hashkey (gensym))
	(hashkey2 (gensym)))
    (setf (gethash hashkey preparations) (make-hash-table :test 'equal))
    (setf (gethash hashkey2 preparations) (make-hash-table :test 'equal))
    
    `(macrolet (,@(loop for i in input collect
			`(,i (&rest rest)
			     (cl:if (cl:eql (cl:length rest) 0)
				    `(abs-array ,',i ,@rest)
				    `(rel-array ,',i ,@rest))))
		
		(set (&rest rest)
		    ;;TODO fuse with sse code
		    (let ((prepare-load (gethash ',hashkey preparations))
			  (prepare-store (gethash ',hashkey2 preparations))
			  (prepared nil))
		      
		      (labels ((find-inputs-outputs (list)
				 (cl:when (listp list)
				   (loop for i in list do
				     (cl:when (listp i)
				       (cl:cond
					 ((find (car i) ',input)
					  (cl:when (cl:not (gethash (butlast i) prepare-load))
					    (setf (gethash (butlast i) prepare-load) (intrinsic-tmp "vec4_"))))
					 ((find (car i) ',output)
					  (cl:when (cl:not (gethash (butlast i) prepare-store))
					    (setf (gethash (butlast i) prepare-store) (intrinsic-tmp "vec4_"))))
					 (t (find-inputs-outputs i))))))))

			(find-inputs-outputs rest)

			(setf prepared `(cms-cuda::set ,@rest))

			(setf prepared
			      `(macrolet ,(loop for type in '(,input ,output)
						for hash in `(,prepare-load ,prepare-store)
						append
						(loop for i in type collect
						      `(,i (&rest rest)
							   (declare (ignorable rest))
							   (cl::cond
							     ,@(loop for k being the hash-keys of hash
								     append
								     (cl::when (equal i (first k))
								       `(((equal rest ',(append (rest k) '(0))) '(oref ',(gethash k hash) 'x))
									 ((equal rest ',(append (rest k) '(1))) '(oref ',(gethash k hash) 'y))
									 ((equal rest ',(append (rest k) '(2))) '(oref ',(gethash k hash) 'z))
									 ((equal rest ',(append (rest k) '(3))) '(oref ',(gethash k hash) 'w)))))))))
				 ,prepared))


			(maphash #'(lambda (key value)
				     (setf prepared `(store-vec4-head ,value ,key ,prepared))) prepare-store)

			(maphash #'(lambda (key value)
				     (setf prepared `(load-vec4-head ,value ,key ,prepared))) prepare-load)
			
			prepared)))
		
		(decl (declarations &body body)
		    (let ((prepare-load (gethash ',hashkey preparations))
			  (prepared `(cms-cuda::decl ,declarations ,@body)))
		      
		      (labels ((find-inputs-outputs (list)
				 (cl:when (listp list)
				   (loop for i in list do
				     (cl:when (listp i)
				       (cl:cond
					 ((find (car i) ',input)
					  (cl:when (cl:not (gethash (butlast i) prepare-load))
					    (setf (gethash (butlast i) prepare-load) (intrinsic-tmp "vec4_"))))
					 (t (find-inputs-outputs i))))))))

			(find-inputs-outputs declarations)

			(setf prepared
			      `(macrolet ,(loop for i in ',input collect
						`(,i (&rest rest)
						     (declare (ignorable rest))
						     (cl::cond
						       ,@(loop for k being the hash-keys of prepare-load
							       append
							       (cl::when (equal i (first k))
								 `(((equal rest ',(append (rest k) '(0))) '(oref ',(gethash k prepare-load) 'x))
								   ((equal rest ',(append (rest k) '(1))) '(oref ',(gethash k prepare-load) 'y))
								   ((equal rest ',(append (rest k) '(2))) '(oref ',(gethash k prepare-load) 'z))
								   ((equal rest ',(append (rest k) '(3))) '(oref ',(gethash k prepare-load) 'w))))))))
				 ,prepared))
			
			(maphash #'(lambda (key value)
				     (setf prepared `(load-vec4-head ,value ,key ,prepared))) prepare-load)
			
			prepared)))) 
       ,@body)))						    


