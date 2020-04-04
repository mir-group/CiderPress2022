module utils
implicit none

real(8), parameter :: pi = 3.141592653589793115997963468544185161590576171875

contains

! module, function, program, or block
! 3xN, not Nx3
function hartree_potential (vh, rho_data, coords, weights, ngrid, ndat)

    implicit none

    integer                                         :: ngrid
    integer                                         :: ndat
    integer                                         :: i
    integer                                         :: j
    integer                                         :: hartree_potential
    real(8), dimension(ndat,ngrid), intent(in)      :: rho_data
    real(8), dimension(3,ngrid), intent(in)         :: coords
    real(8), dimension(ngrid), intent(in)           :: weights
    real(8), dimension(ndat,ngrid), intent(out)     :: vh
    real(8), dimension(3,ngrid)                     :: vecs
    real(8), dimension(ndat,ngrid)                  :: tmp
    real(8), dimension(ngrid)                       :: rs

    hartree_potential = -1
    !$omp parallel do default(shared) private(vecs, rs, tmp)
    do i = 1, ngrid
        do j = 1, ngrid
            vecs(:,j) = coords(:,j) - coords(:,i)
        enddo
        rs(:) = norm2(vecs, 1)
        do j=1, ngrid
            if (rs(j) < 1.0e-6) then
                rs(j) = (2.0/3) * (3.0 * abs(weights(j)) / (4 * pi))**(1.0 / 3)
            endif
        enddo
        rs(i) = (2.0/3) * (3.0 * abs(weights(i)) / (4 * pi))**(1.0 / 3)
        if (weights(i) < 1.0e-10) then
            rs(i) = 1.0e10
        endif
        do j=1, ndat
            tmp(j,:) = rho_data(j,:) / rs
            vh(j,i) = dot_product(tmp(j,:), weights)
            !vh(j,i) = vh(j,i) - rho_data(j,i) / rs(i) * weights(i)
        enddo
    enddo
    !$omp end parallel do
    hartree_potential = 0

end function hartree_potential

function nonlocal_dft_data (nlc_data, rho_data, dtau_data, dvh_data,&
                            ws_radii, coords, weights, ngrid)

    implicit none

    integer                                         :: ngrid
    integer                                         :: i
    integer                                         :: j
    integer                                         :: nonlocal_dft_data
    real(8), dimension(4,ngrid), intent(in)         :: rho_data
    real(8), dimension(3,ngrid), intent(in)         :: dtau_data
    real(8), dimension(3,ngrid), intent(in)         :: dvh_data
    real(8), dimension(ngrid), intent(in)           :: ws_radii
    real(8), dimension(3,ngrid), intent(in)         :: coords
    real(8), dimension(ngrid), intent(in)           :: weights
    real(8), dimension(8,ngrid), intent(out)        :: nlc_data
    real(8), dimension(3,ngrid)                     :: vecs
    real(8), dimension(ngrid)                       :: tmp
    real(8), dimension(ngrid)                       :: rs
    real(8), dimension(ngrid)                       :: exp_weights

    nonlocal_dft_data = -1
    nlc_data(1,:) = norm2(dvh_data(1:3,:))
    !$omp parallel do default(shared) private(vecs, rs, exp_weights, tmp)
    do i = 1, ngrid
        do j = 1, ngrid
            vecs(:,j) = coords(:,j) - coords(:,i)
        enddo
        rs(:) = norm2(vecs, 1)
        exp_weights(:) = exp(-rs / ws_radii(i)) * weights * rho_data(1,:)
        nlc_data(5,i) = sum(exp_weights)
        tmp(:) = sum(vecs * rho_data(2:4,:), 1)
        nlc_data(3,i) = dot_product(tmp, exp_weights)
        tmp(:) = sum(vecs * dtau_data(1:3,:))
        nlc_data(4,i) = dot_product(tmp, exp_weights)
        tmp(:) = sum(vecs * dvh_data(1:3,:))
        nlc_data(2,i) = dot_product(tmp, exp_weights)
        tmp(:) = rho_data(1,:)**(1.0 / 3)
        nlc_data(6,i) = dot_product(tmp, exp_weights)
        tmp(:) = rho_data(1,:)**(5.0 / 3)
        nlc_data(7,i) = dot_product(tmp, exp_weights)
        nlc_data(8,i) = dot_product(rho_data(1,:), exp_weights)
    enddo
    !$omp end parallel do
    nonlocal_dft_data = 0

end function nonlocal_dft_data

function nonlocal_dft_data2 (nlc_data, rho_data, dtau_data, dvh_data,&
                            ws_radii, coords, weights, ngrid)

    implicit none

    integer                                         :: ngrid
    integer                                         :: i
    integer                                         :: j
    integer                                         :: nonlocal_dft_data2
    real(8), dimension(4,ngrid), intent(in)         :: rho_data
    real(8), dimension(3,ngrid), intent(in)         :: dtau_data
    real(8), dimension(3,ngrid), intent(in)         :: dvh_data
    real(8), dimension(ngrid), intent(in)           :: ws_radii
    real(8), dimension(3,ngrid), intent(in)         :: coords
    real(8), dimension(ngrid), intent(in)           :: weights
    real(8), dimension(8,ngrid), intent(out)        :: nlc_data
    real(8), dimension(3,ngrid)                     :: vecs
    real(8), dimension(ngrid)                       :: tmp
    real(8), dimension(ngrid)                       :: rs
    real(8), dimension(ngrid)                       :: exp_weights

    nonlocal_dft_data2 = -1
    nlc_data(1,:) = norm2(dvh_data(1:3,:))
    !$omp parallel do default(shared) private(vecs, rs, exp_weights, tmp)
    do i = 1, ngrid
        do j = 1, ngrid
            vecs(:,j) = coords(:,j) - coords(:,i)
        enddo
        rs(:) = norm2(vecs, 1)
        exp_weights(:) = exp(-rs / ws_radii(i)) * weights * rho_data(1,:)
        nlc_data(5,i) = sum(exp_weights)
        tmp(:) = sum(vecs * rho_data(2:4,:), 1)
        nlc_data(3,i) = dot_product(tmp, exp_weights)
        tmp(:) = sum(vecs * dtau_data(1:3,:))
        nlc_data(4,i) = dot_product(tmp, exp_weights)
        tmp(:) = sum(vecs * dvh_data(1:3,:))
        nlc_data(2,i) = dot_product(tmp, exp_weights)
        tmp(:) = rho_data(1,:)**(1.0 / 3)
        nlc_data(6,i) = dot_product(tmp, exp_weights)
        tmp(:) = rho_data(1,:)**(5.0 / 3)
        nlc_data(7,i) = dot_product(tmp, exp_weights)
        nlc_data(8,i) = dot_product(rho_data(1,:) / rs, exp_weights)
    enddo
    !$omp end parallel do
    nonlocal_dft_data2 = 0

end function nonlocal_dft_data2

function fact (n)

    implicit none
    integer :: n
    integer :: i
    integer :: f 
    real(8) :: fact

    f = 1
    do i=1,n 
        f = f * i
    enddo
    fact = real(f,8)

end function fact

function legendre (l, m, x)

    implicit none

    integer             :: l
    integer             :: m
    integer             :: msign
    integer             :: n
    real(8)             :: x
    real(8)             :: legendre

    msign = sign(1, m)
    m = abs(m)
    n = l
    legendre = 0
    do while ( (n .ge. 0) .and. (2*n-l-m .ge. 0) )
        legendre = legendre &
                    + x**(2*n-l-m) * fact(2*n) / fact(2*n-l-m) &
                    / fact(n) / fact(l-n) * (-1)**(l-n)
        n = n - 1
    enddo
    legendre = legendre * (-1)**m * (1-x*x)**(m/2.0) / 2**l
    if (msign < 0) then
        legendre = (-1)**m * fact(l+m) / fact(l-m) * legendre
    endif

end function legendre

function ylm (l, m, costheta, phi)

    implicit none
    integer             :: l
    integer             :: m
    real(8)             :: costheta ! polar angle
    real(8)             :: phi ! azimuthal angle
    !real(8)             :: pi = 4 * atan(1.0_8)
    real(8)             :: ylm

    if (m.eq.0) then
        ylm = 1.
    else if (m.le.0) then
        m = abs(m)
        ylm = sqrt(2.) * (-1)**m * sin(m * phi)
    else
        ylm = sqrt(2.) * (-1)**m * cos(m * phi)
    endif

    ylm = sqrt((2*l+1) / (4*pi) * fact(l-m) / fact(l+m))&
          * legendre(l, m, costheta) * ylm
end function ylm

function laguerre (n, a, x)

    implicit none
    integer     :: n
    integer     :: a
    integer     :: k
    real(8)     :: x
    real(8)     :: lkm1
    real(8)     :: lkm0
    real(8)     :: laguerre

    laguerre = 0
    if (n.eq.0) then
        laguerre = 1
    else if (n.eq.1) then
        laguerre = 1 + a - x
    else
        lkm1 = 1
        lkm0 = 1 + a - x
        do k=1,n-1
            ! from https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials
            laguerre = ((2*k+1+a-x) * lkm0 - (k+a) * lkm1) / (k+1)
            lkm1 = lkm0
            lkm0 = laguerre
        enddo
    endif
end function laguerre

function hwf (n, l, m, a, r, costheta, phi)

    implicit none
    integer     :: n
    integer     :: l
    integer     :: m
    real(8)     :: r
    real(8)     :: rho
    real(8)     :: a
    real(8)     :: costheta
    real(8)     :: phi
    real(8)     :: hwf

    ! https://en.wikipedia.org/wiki/Hydrogen_atom#Schr%C3%B6dinger_equation
    rho = 2 * r / (n * a)
    hwf = sqrt( (2 / (n * a))**3 * fact(n-l-1) / (2 * n * fact(n+l)) )
    hwf = hwf * exp(-rho/2) * rho**l
    hwf = hwf * laguerre(n-l-1, 2*l+1, rho) * ylm(l, m, costheta, phi)
end function hwf

end module utils
